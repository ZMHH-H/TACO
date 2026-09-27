import os
import sys
import time
import argparse

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
import torchvision
import numpy as np
from utils.utils import init_distributed_mode, epoch_saving, best_saving, AverageMeter, reduce_tensor, accuracy, gen_label, gather_labels, we_saving, cosine_scheduler
from utils.logger import setup_logger
import clip
import copy

from pathlib import Path
import yaml
import pprint
from dotmap import DotMap

import datetime
import shutil
from contextlib import suppress

from modules.video_clip import video_header, VideoCLIP
from utils.NCELoss import NCELoss, DualLoss, BYOLLoss
from utils.Augmentation import get_augmentation
from utils.solver import _optimizer, _lr_scheduler
from modules.text_prompt import text_prompt, text_prompt_ensemble


class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor):
        output = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
        torch.distributed.all_gather(output, tensor)
        ctx.rank = dist.get_rank()
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, dim=0)

    # grad_output: the gradients calculated from the last layer (have same size with the output of forward())
    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank : ctx.batch_size * (ctx.rank + 1)],
            None,
        )

allgather = AllGather.apply

def update_dict(dict):
    new_dict = {}
    for k, v in dict.items():
        new_dict[k.replace('module.', '')] = v
    return new_dict

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', type=str, default='clip.yaml', help='global config file')
    parser.add_argument('--log_time', default='001')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')                        
    parser.add_argument("--local_rank", type=int,
                        help='local rank for DistributedDataParallel')
    parser.add_argument(
        "--precision",
        choices=["amp", "fp16", "fp32"],
        default="amp",
        help="Floating point precition."
    )                        
    args = parser.parse_args()
    return args

@torch.no_grad()
def sample_sphere_gaussian(num_points: int, dim: int, mean=0.0, std=1.0, device="cuda"):
    """
    从高斯分布采样，并投影到单位球面上，可以设置均值和方差

    Args:
        num_points (int): 采样数量
        dim (int): 空间维度（CLIP text space 常用 512）
        mean (float or torch.Tensor): 高斯分布均值，可以是标量或长度为 dim 的向量
        std (float): 高斯分布标准差
        device (str): 'cpu' 或 'cuda'

    Returns:
        torch.Tensor: [num_points, dim] 的单位向量
    """
    # 如果 mean 是标量，转成张量
    if not torch.is_tensor(mean):
        mean = torch.full((dim,), float(mean), device=device)
    else:
        mean = mean.to(device)

    # 标准正太分布采样
    z = torch.randn(num_points, dim, device=device)
    
    # 高斯空间内加std和mean
    z = z * std + mean

    # L2 归一化到球面
    z = z / z.norm(dim=-1, keepdim=True)
    return z



def main(args):
    global best_prec1
    """ Training Program """
    init_distributed_mode(args)
    if args.distributed:
        print('[INFO] turn on distributed train', flush=True)
    else:
        print('[INFO] turn off distributed train', flush=True)

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    working_dir = os.path.join(config['data']['output_path'], config['data']['dataset'], config['network']['arch'] , args.log_time)


    if dist.get_rank() == 0:
        Path(working_dir).mkdir(parents=True, exist_ok=True)
        shutil.copy(args.config, working_dir)
        shutil.copy('train.py', working_dir)
        shutil.copy('./modules/video_clip.py', working_dir)

    # build logger, print env and config
    logger = setup_logger(output=working_dir,
                          distributed_rank=dist.get_rank(),
                          name=f'TaCo')
    logger.info("------------------------------------")
    logger.info("Environment Versions:")
    logger.info("- Python: {}".format(sys.version))
    logger.info("- PyTorch: {}".format(torch.__version__))
    logger.info("- TorchVison: {}".format(torchvision.__version__))
    logger.info("------------------------------------")
    pp = pprint.PrettyPrinter(indent=4)
    logger.info(pp.pformat(config))
    logger.info("------------------------------------")
    logger.info("storing name: {}".format(working_dir))

    config = DotMap(config)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        cudnn.benchmark = True
        # cudnn.deterministic = True

    # fix the seed for reproducibility
    seed = config.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)


    # get fp16 model and weight
    model_clip, clip_state_dict = clip.load(
        config.network.arch,
        device='cpu',jit=False,
        internal_modeling=config.network.tm,
        T=config.data.num_segments,
        dropout=config.network.drop_out,
        emb_dropout=config.network.emb_dropout,
        pretrain=config.network.init,
        joint_st = config.network.joint_st) # Must set jit=False for training  ViT-B/32

    # Data Augmentations
    transform_train = get_augmentation(True, config)
    transform_val = get_augmentation(False, config)
    logger.info('train transforms: {}'.format(transform_train.transforms))
    logger.info('val transforms: {}'.format(transform_val.transforms))

    if args.precision == "amp" or args.precision == "fp32":
        model_clip = model_clip.float()

    if config.data.dataset == 'charades':
        from datasets.charades import Video_dataset
        train_data = Video_dataset(
            config.data.train_root, config.data.train_list,
            config.data.label_list, num_segments=config.data.num_segments,
            modality=config.data.modality,
            image_tmpl=config.data.image_tmpl, random_shift=config.data.random_shift,
            transform=transform_train, dense_sample=config.data.dense,
            fps=config.data.fps)
        val_data = Video_dataset(
            config.data.val_root, config.data.val_list, config.data.label_list,
            random_shift=False, num_segments=config.data.num_segments,
            modality=config.data.modality,
            image_tmpl=config.data.image_tmpl,
            transform=transform_val, test_mode=True, dense_sample=config.data.dense)            
    else:
        from datasets.video import Video_dataset
        train_data = Video_dataset(
            config.data.train_root, config.data.train_list,
            config.data.label_list, num_segments=config.data.num_segments,
            modality=config.data.modality,
            image_tmpl=config.data.image_tmpl, random_shift=config.data.random_shift,
            transform=transform_train, dense_sample=config.data.dense,
            train_clips=1)
        val_data = Video_dataset(
            config.data.val_root, config.data.val_list, config.data.label_list,
            random_shift=False, num_segments=config.data.num_segments_val,
            modality=config.data.modality,
            image_tmpl=config.data.image_tmpl,
            transform=transform_val, dense_sample=config.data.dense)

    ################ Few-shot data for training ###########
    if config.data.shot:
        cls_dict = {}
        for item  in train_data.video_list:
            if item.label not in cls_dict:
                cls_dict[item.label] = [item]
            else:
                cls_dict[item.label].append(item)
        import random
        # seed must be same across GPUs, otherwise different videos will be sampled at each GPU
        random.seed(config.seed) 
        select_vids = []
        K = config.data.shot
        for category, v in cls_dict.items():
            slice = random.sample(v, K)
            select_vids.extend(slice)
            # for s in slice:
            #     print('slice', s.path, '/n')
        n_repeat = len(train_data.video_list) // len(select_vids)
        train_data.video_list = select_vids * n_repeat
        # print('########### number of videos: {} #########'.format(len(select_vids)))
    ########################################################


    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)  # set seed=seed to change the data order
    train_loader = DataLoader(train_data,
        batch_size=config.data.batch_size, num_workers=config.data.workers,
        sampler=train_sampler, drop_last=False)

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data, shuffle=False)
    val_loader = DataLoader(val_data,
        batch_size=config.data.batch_size,num_workers=config.data.workers,
        sampler=val_sampler, drop_last=False)

    loss_type = config.solver.loss_type
    if loss_type == 'CE':
        print('============= Using CE Loss ==============')
        criterion = torch.nn.CrossEntropyLoss()
    elif loss_type == 'NCE':
        print('============= Using NCE Loss ==============')
        criterion = NCELoss()
    else:
        raise NotImplementedError

    # # ============= generate class features ==============
    # print('============= Start encoding class features ===========')
    # classes = text_prompt_ensemble(train_data)
    # # n_class = classes[0].size(0)
    # model_clip.cuda()
    # model_clip.eval()
    # with torch.no_grad():
    #     # @zmhh_h multi text prompts
    #     cls_feature_list = [model_clip.encode_text(classes[i].cuda(), return_token=True)[0] for i in range(len(classes))]
    #     for cls_feature in cls_feature_list:
    #         cls_feature /= cls_feature.norm(dim=-1, keepdim=True)
    #     cls_feature = torch.stack(cls_feature_list, 0).mean(0)
    #     cls_feature /= cls_feature.norm(dim=-1, keepdim=True)
    # print('============= End encoding class features ===========')
    

    model = VideoCLIP(model_clip, config.data.num_segments)
    del model_clip

    # initialize teacher
    model_teacher = copy.deepcopy(model) # deepcopy

    # Temporal Aggregation Module
    video_head = video_header(
        config.network.sim_header,
        config.network.interaction,
        clip_state_dict,
        config.network.temporal_layer,
        topk_frame = config.network.topk_frame,
        teacher_momentum = config.network.teacher_momentum)
    
    start_epoch = config.solver.start_epoch
    
    if config.pretrain:
        if os.path.isfile(config.pretrain):
            logger.info("=> loading checkpoint '{}'".format(config.pretrain))
            checkpoint = torch.load(config.pretrain, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            video_head.load_state_dict(checkpoint['fusion_model_state_dict'])
            del checkpoint
        else:
            logger.info("=> no checkpoint found at '{}'".format(config.resume))
    
    if config.resume:
        if os.path.isfile(config.resume):
            logger.info("=> loading checkpoint '{}'".format(config.resume))
            checkpoint = torch.load(config.resume, map_location='cpu')
            model.load_state_dict(update_dict(checkpoint['model_state_dict']))
            video_head.load_state_dict(update_dict(checkpoint['fusion_model_state_dict']))
            start_epoch = checkpoint['epoch'] + 1
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(config.evaluate, checkpoint['epoch']))
            del checkpoint
        else:
            logger.info("=> no checkpoint found at '{}'".format(config.pretrain))

    # tokenized specific text prompts
    # classes: [number_classes, 77]
    # classes: dict:(num_template : [num_cls, 77])
    classes = text_prompt_ensemble(train_data)
    # n_class = classes[0].size(0)

    if config.network.fix_text:
        for name, param in model.named_parameters():
            if "visual" not in name and "logit_scale" not in name:
                param.requires_grad_(False)

    if config.network.fix_video:
        for name, param in model.named_parameters():
            if "visual" in name:
                param.requires_grad_(False)
    
    # ============== set optimizer ==============
    optimizer = _optimizer(config, model, video_head)
    lr_scheduler = _lr_scheduler(config, optimizer)

    if args.distributed:
        model = DistributedDataParallel(model.cuda(), device_ids=[args.gpu])
        model_teacher = DistributedDataParallel(model_teacher.cuda(), device_ids=[args.gpu])
        if config.network.sim_header in ["None","Selective"] and config.network.interaction in ['DP']:
            video_head_nomodule = video_head
        else:
            video_head = DistributedDataParallel(video_head.cuda(), device_ids=[args.gpu], find_unused_parameters=False)
            video_head_nomodule = video_head.module

    scaler = GradScaler() if args.precision == "amp" else None
    
    best_prec1 = 0.0

    # weight ensemble
    we_model =      {key: value.cpu().clone() for key, value in model.module.state_dict().items()}
    we_video_head = {key: value.cpu().clone() for key, value in video_head_nomodule.state_dict().items()}
    
    momentum_scheduler = cosine_scheduler(config.network.teacher_momentum, 1.0,
                                            config.solver.epochs, len(train_loader))

    if config.solver.evaluate:
        logger.info(("===========evaluate==========="))
        # TODO: Add charades validation function
        prec1 = validate(start_epoch, val_loader, device, model, video_head, config, classes, logger)
        return

    for epoch in range(start_epoch, config.solver.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)        
        
        train(model, video_head, train_loader, optimizer, criterion, scaler,
              epoch, device, lr_scheduler, config, classes, logger, model_teacher, momentum_scheduler)

        # weight ensemble
        if dist.get_rank() == 0:
            logger.info('Weight Ensemble Saving: {}/we_model.pt'.format(working_dir))
            we_model, we_video_head = we_saving(epoch, model.module, we_model, video_head_nomodule, 
                                                    we_video_head, "{}/we_model.pt".format(working_dir))

            logger.info('Teacher Model Saving: {}/teacher_model.pt'.format(working_dir))
            epoch_saving(epoch, model_teacher.module, video_head_nomodule, optimizer, "{}/teacher_model.pt".format(working_dir))
        
        if (epoch+1) % config.logging.eval_freq == 0:  # and epoch>0
            prec1 = validate(epoch, val_loader, device, model, video_head, config, classes, logger)

            if dist.get_rank() == 0:
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                logger.info('Testing: {}/{}'.format(prec1,best_prec1))
                logger.info('Saving:')
                filename = "{}/last_model.pt".format(working_dir)

                epoch_saving(epoch, model.module, video_head_nomodule, optimizer, filename)
                if is_best:
                    best_saving(working_dir, epoch, model.module, video_head_nomodule, optimizer)


def train(model, video_head, train_loader, optimizer, criterion, scaler,
          epoch, device, lr_scheduler, config, classes, logger, model_teacher, momentum_scheduler):
    """ train a epoch """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    img_losses = AverageMeter()
    text_losses = AverageMeter()

    model.train()
    video_head.train()
    model_teacher.eval()
    
    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress
    end = time.time()

    for i,(data, list_id) in enumerate(train_loader):
        if config.solver.type != 'monitor':
            if (i + 1) == 1 or (i + 1) % 10 == 0:
                lr_scheduler.step(epoch + i / len(train_loader))
        # lr_scheduler.step()

        data_time.update(time.time() - end)

        # # ============== train_clips = 2 ===============
        # images, images_tgt = data
        # list_id=list_id[0]

        # images = images.view((-1, config.data.num_segments, 3) + images.size()[-2:])  # b t 3 h w
        # images_tgt = images_tgt.view((-1, config.data.num_segments, 3) + images.size()[-2:])  # b t 3 h w
        # # # ============ visualize frames ==========
        # # import utils.visualization
        # # for i in range(images.shape[0]):
        # #     utils.visualization.show_video_frame(images[i].permute(1,0,2,3),window_name='111')
        # #     utils.visualization.show_video_frame(images_tgt[i].permute(1,0,2,3),window_name='222')
        # # # ============ end visualize =============
        # b,t,c,h,w = images.size()
        # # omit the Image.fromarray if the images already in PIL format, change this line to images=list_image if using preprocess inside the dataset class
        # images = images.view(-1, c, h, w)
        # images_tgt = images_tgt.view(-1, c, h, w)
        
        # ============== train_clips = 1 ===============
        images = data
        list_id = list_id
        images = images.view((-1, config.data.num_segments, 3) + images.size()[-2:])  # b t 3 h w

        # # ============ visualize frames ==========
        # import utils.visualization
        # for i in range(images.shape[0]):
        #     utils.visualization.show_video_frame(images[i].permute(1,0,2,3),window_name='aaa')
        # # ============ end visualize =============
        b,t,c,h,w = images.size()
        images = images.view(-1, c, h, w)

        texts = classes
        
        beta = config.network.teacher_momentum    # FIXED Momentum
        # global_steps = len(train_loader) * epoch + i
        # beta = momentum_scheduler[global_steps]

        with autocast():
            if config.solver.loss_type in ['CE']:
                # image_embedding:  [BS, T, C]
                # text_embedding:   [num_cls, C]
                # logit_scale:      [1/0.07] used to scale the logits
                image_embedding, text_embedding, logit_scale = model(images, texts)
                
                # # raw logits
                # img_emb_raw = image_embedding.mean(dim=1, keepdim=False)
                # text_emb_raw = text_embedding
                # img_emb_raw = img_emb_raw / img_emb_raw.norm(dim=-1, keepdim=True) # [B, C]
                # text_emb_raw = text_emb_raw / text_emb_raw.norm(dim=-1, keepdim=True) # [N_cls, C]
                # logits_raw = img_emb_raw @ text_emb_raw.t() # [B, N_cls]

                # sample ood text embeddings [N_points, C]
                ood_embbeding = sample_sphere_gaussian(num_points = 200, dim = image_embedding.size(-1), mean=0.0, std=1.0)
                
                with torch.no_grad():
                    # EMA update for teacher model
                    for (name_s, param_s), (name_t, param_t) in zip(model.module.named_parameters(), model_teacher.module.named_parameters()):
                        param_t.data = param_t.data * beta + param_s.data * (1. - beta)
                    # teacher model forward
                    img_emb_teacher, text_emb_teacher, logit_scale_teacher = model_teacher(images, texts)
                    
                    # teacher logits
                    img_emb_teacher = img_emb_teacher.mean(dim=1, keepdim=False)
                    img_emb_teacher = img_emb_teacher / img_emb_teacher.norm(dim=-1, keepdim=True) # [B, C]
                    text_emb_teacher = text_emb_teacher / text_emb_teacher.norm(dim=-1, keepdim=True) # [N_cls, C]
                    # logits_teacher = img_emb_teacher @ text_emb_teacher.t() # [B, N_cls]
                    logits_teacher = (img_emb_teacher @ ood_embbeding.t()) @ (ood_embbeding @ text_emb_teacher.t()) # [B, N_cls]

                # logits, proj_vid, proj_text, proj_vid_teacher, proj_text_teacher = video_head(image_embedding, text_embedding, img_emb_teacher, text_emb_teacher)
                # used for w.o. projector and predictor
                img_emb_proj, logits = video_head(image_embedding, text_embedding, img_emb_teacher, text_emb_teacher)

                loss_ce = criterion(logit_scale * logits, list_id.to(device))

                # # hybrid loss
                # one_hot_id = F.one_hot(list_id.to(device), num_classes = logits.shape[-1]).float() # [B, N_cls]
                # teacher_id = F.softmax(logits_teacher / 0.01, dim=-1) # [B, N_cls]
                # # print('teacher_id', teacher_id)
                # alpha = 0.7
                # hybrid_id = alpha * one_hot_id + (1-alpha) * teacher_id
                # loss_ce = criterion(logit_scale * logits, hybrid_id)
                # # print('loss_hybrid',loss_ce)

                # BYOL Loss
                # loss_vid_distill = BYOLLoss(image_embedding, img_emb_teacher)
                # loss_text_distill = BYOLLoss(text_embedding, text_emb_teacher)
                # loss = loss_ce + 0.5*loss_vid_distill + 0.5*loss_text_distill

                # Calculate Distillation Logits
                img_emb_proj = img_emb_proj.mean(dim=1, keepdim=False)
                img_emb_proj = img_emb_proj / img_emb_proj.norm(dim=-1, keepdim=True) # [B, C]
                text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True) # [B, C]
                logits_stu = (img_emb_proj @ ood_embbeding.t()) @ (ood_embbeding @ text_embedding.t()) # [B, N_cls]
                # print('logits_stu', logits_stu.shape)

                # KL Loss
                ref_temperature = 1.0
                log_logits_student = F.log_softmax(logit_scale * logits_stu / ref_temperature, dim=-1)
                logits_teacher = F.softmax(logit_scale * logits_teacher / ref_temperature, dim=-1)
                loss_kl = F.kl_div(log_logits_student, logits_teacher, reduction='batchmean')*(ref_temperature**2)
                loss = loss_ce + 0.4*loss_kl
                # loss = 0.7*loss_ce + 0.3*loss_kl

                # # ============ visualize frames ==========
                # import utils.visualization
                # images_ = images.reshape(b,t,c,h,w)
                # for i in range(images_.shape[0]):
                #     utils.visualization.show_video_frame(images_[i].permute(1,0,2,3),window_name='aaa')
                # # ============ end visualize =============
                
            elif config.solver.loss_type in ['NCE']:
                # create batch text embeddings
                # text_embedding_batch=text_embedding[list_id]
                batch_texts = {key: value[list_id] for key, value in texts.items()}

                # image_embedding: [BS, T, C]
                # cls_embedding: [BS, C]
                # logit_scale: [1/0.07] used to scale the logits
                image_embedding, text_embedding, logit_scale= model(images, batch_texts)

                # allgather text embeddings before video_head
                # image_embedding = allgather(image_embedding)
                text_embedding = allgather(text_embedding)

                # video_head for temporal modeling
                _, logits = video_head(image_embedding, text_embedding)
                logits = logit_scale * logits

                # all gather logits after video_head (need to allgather text embeddings before video_head)
                logits = allgather(logits)
                
                # generate gt matrix
                # gt = [bs bs]
                list_id = gather_labels(list_id.to(device))  # bs -> n_gpu * bs
                ground_truth = torch.tensor(gen_label(list_id),dtype=image_embedding.dtype, device=device)
                # print('gt Matrix:', ground_truth.shape, ground_truth)
                
                loss_nce_img = criterion(logits, ground_truth)
                loss_nce_txt = criterion(logits.T, ground_truth)
                loss_nce = (loss_nce_img+loss_nce_txt)/2
                loss = loss_nce
            else:
                raise NotImplementedError
            
            # loss regularization
            # 因为这里的loss的.grad已经累加了grad_accumulation_steps次，实际上约等于将梯度值放大了grad_accumulation_steps倍
            # 所以要将loss放缩回来
            loss = loss / config.solver.grad_accumulation_steps

        if scaler is not None:
            # back propagation
            scaler.scale(loss).backward()
            if (i + 1) % config.solver.grad_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()  # reset gradient
        else:
            # back propagation
            loss.backward()
            if (i + 1) % config.solver.grad_accumulation_steps == 0:
                optimizer.step()  # update param
                optimizer.zero_grad()  # reset gradient

        losses.update(loss.item(), logits.size(0))


        batch_time.update(time.time() - end)
        end = time.time()
        cur_iter = epoch * len(train_loader) + i
        max_iter = config.solver.epochs * len(train_loader)
        eta_sec = batch_time.avg * (max_iter - cur_iter + 1)
        eta_sec = str(datetime.timedelta(seconds=int(eta_sec)))

        if i % config.logging.print_freq == 0:
            logger.info(('Epoch: [{0}][{1}/{2}], lr: {lr:.2e}, eta: {3}\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                             epoch, i, len(train_loader), eta_sec, batch_time=batch_time, data_time=data_time, loss=losses,
                             lr=optimizer.param_groups[-1]['lr'])))




def validate(epoch, val_loader, device, model, video_head, config, classes, logger):
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    video_head.eval()

    with torch.no_grad():
        text = classes
        text_embedding_list = [model.module.encode_text(text[i].cuda(), return_token=False)[0] for i in range(len(text))]
        # mean(0) first of last?? 
        text_embedding = torch.stack(text_embedding_list, 0).mean(0) # [num_cls, C]
        text_embedding /= text_embedding.norm(dim=-1, keepdim=True)  # [num_cls, C]
        
        for i, (image, class_id) in enumerate(val_loader):
            image = image.view((-1, config.data.num_segments_val, 3) + image.size()[-2:])
            b, t, c, h, w = image.size()
            class_id = class_id.to(device)
            image = image.to(device).view(-1, c, h, w) # [BS*T, C, H, W]

            image_embedding = model.module.encode_image(image, config.data.num_segments_val) # [BS, T, C]
            similarity = video_head(image_embedding, text_embedding) # [BS, n_cls]


            prec = accuracy(similarity, class_id, topk=(1, 5))
            prec1 = reduce_tensor(prec[0])
            prec5 = reduce_tensor(prec[1])

            top1.update(prec1.item(), class_id.size(0))
            top5.update(prec5.item(), class_id.size(0))

            if i % config.logging.print_freq == 0:
                logger.info(
                    ('Test: [{0}/{1}]\t'
                     'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                     'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                         i, len(val_loader), top1=top1, top5=top5)))
    logger.info(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
        .format(top1=top1, top5=top5)))
    return top1.avg




if __name__ == '__main__':
    args = get_parser() 
    main(args)

