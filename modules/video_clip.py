import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.utils.checkpoint import checkpoint
import math
# torch.set_printoptions(threshold=float('inf'))
import copy
# import os

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x



class TemporalTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
        self.grad_checkpointing = False

    def forward(self, x: torch.Tensor):
        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x)
            else:
                x = r(x)
        return x


class video_header(nn.Module):
    def __init__(self, vid_head, interaction, clip_state_dict, temporal_layer=3, topk_frame=8, teacher_momentum=0.9996):
        super().__init__()
        self.vid_header = vid_head
        self.interaction = interaction
        assert vid_head in ["None", "Transf", "Selective", "Self-Distill", "vision_proj"]

        if self.vid_header == "Transf":
            embed_dim = clip_state_dict["text_projection"].shape[1]

            context_length = clip_state_dict["positional_embedding"].shape[0]
            vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
            transformer_width = clip_state_dict["ln_final.weight"].shape[0]
            transformer_heads = transformer_width // 64

            transformer_layers = len(
                set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))

            self.frame_position_embeddings = nn.Embedding(context_length, embed_dim)

            self.transformer = TemporalTransformer(width=embed_dim, layers=temporal_layer, heads=transformer_heads)
            print('=============== num temporal transformer layer: ',temporal_layer, '===============')

            self.apply(self.init_weights)

        elif self.vid_header == "vision_proj":
            # ================= text projection ====================
            embed_dim = clip_state_dict["text_projection"].shape[1]
            self.vision_proj = MLP(embed_dim, hidden_dim = embed_dim*2)
            # self.apply(self.init_weights)

        elif self.vid_header == "Selective":
            self.topk_frame = topk_frame

        elif self.vid_header == "Self-Distill":
            self.beta = teacher_momentum
            MLP_dim = clip_state_dict["visual.proj"].shape[1]
            # vision projection
            self.projector_vid = MLP(MLP_dim, hidden_dim=2048)
            self.projector_vid_teacher = copy.deepcopy(self.projector_vid)
            for name, param in self.projector_vid_teacher.named_parameters():   # set requires_grad to false
                param.requires_grad = False

            # text projection
            self.projector_text = MLP(MLP_dim, hidden_dim=2048)
            self.projector_text_teacher = copy.deepcopy(self.projector_text)
            for name, param in self.projector_text_teacher.named_parameters():   # set requires_grad to false
                param.requires_grad = False
            # self.predictor = MLP(MLP_dim, hidden_dim=2048)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def agg_video_feat(self, x):
        b, t, c = x.size()
        x = x.contiguous()
        if self.vid_header == "None":
            pass

        elif self.vid_header == "Transf":
            x_original = x
            seq_length = t
            position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(0).expand(x.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            x = x + frame_position_embeddings

            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = x.type(x_original.dtype) + x_original
        
        elif self.vid_header == "vision_proj": 
            x_proj = self.vision_proj(x.mean(dim=1, keepdim=True)) # project_vid_feat
            # x_proj = self.vision_proj(x) # project_img_feat
            x = x + x_proj
        else:
            raise ValueError('Unknown temporal modeling header: {}'.format(self.vid_header))
        return x


    def get_logits(self, vid_emb, cls_emb):
        if self.interaction == 'DP':
            vid_emb = vid_emb.mean(dim=1, keepdim=False)  # b c
            vid_emb = vid_emb / vid_emb.norm(dim=-1, keepdim=True)
            cls_emb = cls_emb / cls_emb.norm(dim=-1, keepdim=True)
            logit = vid_emb @ cls_emb.t()  
            # print('final prediction',logit.topk(5, -1)[1])
        else:
            raise NotImplementedError
        # elif self.interaction == 'VCS':  # video concept spotting
        #     cls_emb = cls_emb / cls_emb.norm(dim=-1, keepdim=True)
        #     vid_emb = vid_emb / vid_emb.norm(dim=-1, keepdim=True)
        #     text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        #     sims = torch.einsum('awd,btd->abwt', [text_emb, vid_emb])
        #     att_weight_v = F.softmax(sims/0.01, dim=-1) # abwt
        #     att_weight_v = att_weight_v.mean(dim=-2)  # abt
        #     v_att = torch.einsum('abt,btd->abd', [att_weight_v, vid_emb])
        #     # new
        #     t2v_logits = torch.einsum('abd,ad->ab',[v_att, cls_emb])

        #     logit = t2v_logits.transpose(1, 0)
        
        return logit
    
    def self_distill_modeling(self, vid_emb, cls_emb, vid_teacher, text_teacher):
        proj_vid = self.projector_vid(vid_emb.mean(1))    # [BS, C]
        proj_text = self.projector_text(cls_emb)    # [BS, C]

        with torch.no_grad():
            # EMA update for vision encoder
            for param_s, param_t in zip(self.projector_vid.parameters(), self.projector_vid_teacher.parameters()):
                param_t.data = param_t.data * self.beta + param_s.data * (1. - self.beta)  
            
            # EMA update for text encoder
            for param_s, param_t in zip(self.projector_text.parameters(), self.projector_text_teacher.parameters()):
                param_t.data = param_t.data * self.beta + param_s.data * (1. - self.beta)
            
            proj_vid_teacher = self.projector_vid_teacher(vid_teacher.mean(1))    # [BS, C]
            proj_text_teacher = self.projector_text_teacher(text_teacher)    # [BS, C]
        return proj_vid, proj_text, proj_vid_teacher, proj_text_teacher

    def selective_modeling(self, vid_emb, cls_emb, ref_emb, ref_text_emb, list_id):
        if self.vid_header == "Selective":
            with torch.no_grad():
                vid_tmp = vid_emb / vid_emb.norm(dim=-1, keepdim=True)  # [BS, T, C]
                cls_tmp = cls_emb / cls_emb.norm(dim=-1, keepdim=True)  # [n_cls, C]
                ref_tmp = ref_emb / ref_emb.norm(dim=-1, keepdim=True)  # [BS, T, C]
                
                logit_pred = vid_tmp @ cls_tmp.t()  # [BS, T, n_cls]
                logit_ref = ref_tmp @ cls_tmp.t()   # [BS, T, n_cls]
                # logit_ref = ref_tmp @ ref_text_emb.t()   # [BS, T, n_cls]
                BS, T, N_CLS = logit_pred.shape
                # print('prediction',logit_pred.mean(1).topk(5, -1)[1])
                # print('list_id',list_id)            # 确认一下DDP中每个device的list_id是否正常

                # === entropy of the entire distribution ===
                # 计算整个logits分布的entropy， 不需要label
                # entropy_dist     = -(F.softmax(logit_pred, dim=-1) * F.log_softmax(logit_pred, dim=-1)).sum(-1) # [BS, T]
                # entropy_dist_ref = -(F.softmax(logit_ref, dim=-1) * F.log_softmax(logit_ref, dim=-1)).sum(-1) # [BS, T]
                # entropy_diff = entropy_dist-entropy_dist_ref
                
                # === entropy of the label prediction ===
                # 计算label对应类别prediction的entropy, 相当于使用one-hot label计算分布的熵
                id_expanded = list_id.reshape(-1,1,1).expand(BS, T, 1)
                selected_pred = torch.gather(logit_pred, dim=-1, index=id_expanded).squeeze(-1) # [BS, T]
                # print('selected_pred',selected_pred)
                entropy_pred = -torch.log(selected_pred) # [BS, T]
                # print('entropy_pred',entropy_pred)

                selected_ref = torch.gather(logit_ref, dim=-1, index=id_expanded).squeeze(-1)   # [BS, T]
                # print('selected_ref',selected_ref)
                entropy_ref = -torch.log(selected_ref) # [BS, T]
                # print('entropy_ref',entropy_ref)
                # print('meannnnnn', entropy_pred.mean(-1), entropy_ref.mean(-1))
                # print('Difference',entropy_pred-entropy_ref)
                entropy_diff = entropy_pred-entropy_ref # [BS, T]
                
                # # ===== 1. Generate Reference Mask =====
                # # relative_pred = torch.le(entropy_pred, entropy_pred.mean(-1, True))
                # # print('relative_predddddd','\n',relative_pred)
                # mask = torch.le(entropy_ref, entropy_ref.mean(-1, True)) # [BS, T]
                # f_count = mask.sum(-1, True).unsqueeze(-1) # [BS, 1, 1]
                # # print('relative_maskkkkkk','\n', mask)
                # # print('f_count', f_count)
                # mask = mask.unsqueeze(-1).expand(-1, -1, vid_emb.shape[-1]) # [BS, T, C]
                
                # ===== 2. Generate TopK Mask =====
                _, topk_indices = torch.topk(entropy_diff, self.topk_frame, dim=-1)
                mask = torch.zeros_like(entropy_diff, dtype=torch.bool)
                mask.scatter_(dim=-1, index=topk_indices,value=True)
                # print('mask:','\n',mask)
                mask = mask.unsqueeze(-1).expand(-1, -1, vid_emb.shape[-1]) # [BS, T, C]

                # # ===== 3. Generate TopK Ref Mask =====
                # _, topk_indices = torch.topk(-entropy_ref,self.topk_frame,dim=-1)
                # mask = torch.zeros_like(entropy_ref, dtype=torch.bool)
                # mask.scatter_(dim=-1, index=topk_indices,value=True)
                # # print('mask:','\n',mask)
                # mask = mask.unsqueeze(-1).expand(-1, -1, vid_emb.shape[-1]) # [BS, T, C]
            
            # # ========== 1. ===========
            # vid_emb = (vid_emb*mask).sum(1, keepdim=True)/f_count # [BS, 1, C]
            # # print('vid_emb',vid_emb.shape)
            
            # ========== 2. 3. ===========
            BS, T_, C = mask.shape
            vid_emb = vid_emb[mask].reshape(BS, -1, C) # [BS, T_, C]
            # print('vid_emb',vid_emb.shape)

            return vid_emb
        else:
            raise ValueError('Unknown selective modeling header: {}'.format(self.vid_header))

    def forward(self, vid_emb, cls_emb, ref_emb=None, ref_text_emb=None, list_id=None):
        if self.training:
            if self.vid_header in ["None", "Transf"]:
                vid_emb = self.agg_video_feat(vid_emb)
                logits = self.get_logits(vid_emb, cls_emb)
                return vid_emb, logits
            elif self.vid_header == "vision_proj":
                vid_emb = self.agg_video_feat(vid_emb)
                logits = self.get_logits(vid_emb, cls_emb)
                return vid_emb, logits
            
            elif self.vid_header == "Selective":
                vid_emb = self.selective_modeling(vid_emb, cls_emb, ref_emb, ref_text_emb, list_id)
            elif self.vid_header == "Self-Distill":
                logits = self.get_logits(vid_emb, cls_emb)
                proj_vid, proj_text, proj_vid_teacher, proj_text_teacher = self.self_distill_modeling(vid_emb, cls_emb, ref_emb, ref_text_emb)
                return logits, proj_vid, proj_text, proj_vid_teacher, proj_text_teacher 
            else:
                raise ValueError('Unknown temporal modeling header: {}'.format(self.vid_header))
            
            logits = self.get_logits(vid_emb, cls_emb)
            
            return logits
        else:
            if self.vid_header in ["None", "Transf", "vision_proj"]:
                vid_emb = self.agg_video_feat(vid_emb)
            elif self.vid_header == "Selective":
                vid_emb = self.selective_modeling(vid_emb, cls_emb, ref_emb, ref_text_emb, list_id)
            logits = self.get_logits(vid_emb, cls_emb)
            return logits

class VideoCLIP(nn.Module):
    def __init__(self, clip_model, n_seg, text_encoder = True) :
        super(VideoCLIP, self).__init__()
        # visual encoder parameters
        self.visual = clip_model.visual
        self.n_seg = n_seg
        self.logit_scale = clip_model.logit_scale
        
        if text_encoder:
            # text encoder parameters
            self.transformer = clip_model.transformer # text encoder
            self.token_embedding = clip_model.token_embedding
            self.dtype = clip_model.dtype
            self.positional_embedding = clip_model.positional_embedding
            self.emb_dropout = clip_model.emb_dropout
            self.dropout = clip_model.dropout
            self.ln_final = clip_model.ln_final
            self.text_projection = clip_model.text_projection

    def forward(self, image, text, return_token=False):
        # CLIP encode images
        image_emb = self.encode_image(image) # [BS, T, C]

        # CLIP encode text dict
        # cls_feat, text_feats = self.encode_text(text, return_token)
        cls_feature_list = [self.encode_text(text[i].cuda(), return_token=False)[0] for i in range(len(text))]

        # mean(0) first or last??
        cls_feature = torch.stack(cls_feature_list, 0).mean(0) # [num_templates, num_classes, C]
        cls_feature = cls_feature / cls_feature.norm(dim=-1, keepdim=True)
        # cls_feature = cls_feature.mean(0)

        return image_emb, cls_feature, self.logit_scale.exp()

    def encode_image(self, image, n_segments=None):
        n_seg = self.n_seg if n_segments is None else n_segments
        bt = image.size(0) # [BS*T, C, H, W]
        b = bt // n_seg
        image_emb = self.visual(image) # [BS*T, C]
        image_emb = image_emb.view(b, n_seg, -1) # [BS, T, C]
        return image_emb

    def encode_text(self, text, return_token=False):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        if self.emb_dropout > 0:
            x = self.dropout(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)  # eg, [400 77 512]

        text_token = x @ self.text_projection   # eg, [400 77 512]

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection   # 400 512 

        if return_token:
            return x, text_token
        else:
            return x, None


# class MLP(nn.Module):
#     def __init__(self, emb_dim, hidden_dim=2048):
#         super(MLP, self).__init__()
#         self.linear_A = nn.Linear(emb_dim, hidden_dim)
#         # self.activation = nn.ReLU()
#         self.linear_B = nn.Linear(hidden_dim, emb_dim)

#         nn.init.kaiming_uniform_(self.linear_A.weight, a=math.sqrt(5))
#         # nn.init.zeros_(self.linear_A.weight)
#         # nn.init.zeros_(self.linear_A.bias)
#         nn.init.zeros_(self.linear_B.weight)
#         nn.init.zeros_(self.linear_B.bias)

#     def forward(self, x):
#         # x: [BS, C]
#         x = self.linear_A(x)
#         # x = self.activation(x)
#         x = self.linear_B(x)

#         return x

# class MLP(nn.Module):
#     def __init__(self, emb_dim, hidden_dim=2048):
#         super(MLP, self).__init__()
#         self.linear_A = nn.Linear(emb_dim, emb_dim)
        
#         nn.init.zeros_(self.linear_A.weight)
#         nn.init.zeros_(self.linear_A.bias)
#         # nn.init.kaiming_uniform_(self.linear_A.weight, a=math.sqrt(5))

#     def forward(self, x):
#         # x: [BS, C]
#         x = self.linear_A(x)

#         return x

class MLP(nn.Module):
    def __init__(self, emb_dim, hidden_dim=32):
        super(MLP, self).__init__()
        self.linear_A = nn.Linear(emb_dim, hidden_dim)
        # self.activation = nn.ReLU()
        self.linear_B = nn.Linear(hidden_dim, emb_dim)

        nn.init.kaiming_uniform_(self.linear_A.weight, a=math.sqrt(5))
        # nn.init.zeros_(self.linear_A.weight)
        # nn.init.zeros_(self.linear_A.bias)
        nn.init.zeros_(self.linear_B.weight)
        nn.init.zeros_(self.linear_B.bias)

    def forward(self, x):
        # x: [BS, C]
        x = self.linear_A(x)
        # x = self.activation(x)
        x = self.linear_B(x)

        return x