import os
import sys
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
import time
from utils.utils import init_distributed_mode, AverageMeter, reduce_tensor, accuracy
from utils.logger import setup_logger
import clip
import numpy as np

import yaml
import pprint
from dotmap import DotMap
from datasets.video import Video_dataset
from datasets.transforms import GroupScale, GroupCenterCrop, Stack, ToTorchFormatTensor, GroupNormalize, GroupOverSample, GroupFullResSample
from modules.video_clip import video_header, VideoCLIP
from modules.text_prompt import text_prompt, text_prompt_ensemble, text_prompt_multi


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='global config file')
    parser.add_argument('--weights', type=str, default=None)
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
    parser.add_argument('--test_crops', type=int, default=1)   
    parser.add_argument('--test_clips', type=int, default=1) 
    parser.add_argument('--dense', default=False, action="store_true",
                    help='use dense sample for test as in Non-local I3D')
    args = parser.parse_args()
    return args

def update_dict(dict):
    new_dict = {}
    for k, v in dict.items():
        new_dict[k.replace('module.', '')] = v
    return new_dict


def main(args):
    init_distributed_mode(args)

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = DotMap(config)

    # build logger, print env and config
    logger = setup_logger(output='./zs_eval.txt',
                          distributed_rank=dist.get_rank(),
                          name=f'MoTE')
    logger.info("------------------------------------")
    logger.info("Environment Versions:")
    logger.info("- Python: {}".format(sys.version))
    logger.info("- PyTorch: {}".format(torch.__version__))
    logger.info("- TorchVison: {}".format(torchvision.__version__))
    logger.info("------------------------------------")
    pp = pprint.PrettyPrinter(indent=4)
    logger.info(pp.pformat(config))
    logger.info("------------------------------------")

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        cudnn.benchmark = True

    # get fp16 model and weight
    model_clip, clip_state_dict = clip.load(
        config.network.arch,
        device='cpu', jit=False,
        internal_modeling=config.network.tm,
        T=config.data.num_segments,
        dropout=config.network.drop_out,
        emb_dropout=config.network.emb_dropout,
        pretrain=config.network.init,
        joint_st= config.network.joint_st) # Must set jit=False for training  ViT-B/32


    if args.precision == "amp" or args.precision == "fp32":
        model_clip = model_clip.float()


    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]

    # # rescale size
    # if 'something' in config.data.dataset:
    #     scale_size = (240, 320)
    # else:
    #     scale_size = 224 if config.data.input_size == 224 else config.data.input_size
    # rescale to the size of the full image
    scale_size = int(config.data.input_size)

    # crop size
    input_size = config.data.input_size

    # control the spatial crop
    if args.test_crops == 1: # one center crop
        cropping = torchvision.transforms.Compose([
            GroupScale(scale_size),
            GroupCenterCrop(input_size),
        ])
    elif args.test_crops == 3:  # do not flip, so only 3 crops (left right center)
        cropping = torchvision.transforms.Compose([
            GroupFullResSample(
                crop_size=input_size,
                scale_size=scale_size,
                flip=False)
        ])
    elif args.test_crops == 5:  # do not flip, so only 5 crops (upper left, upper right, lower right, lower left, center)
        cropping = torchvision.transforms.Compose([
            GroupOverSample(
                crop_size=input_size,
                scale_size=scale_size,
                flip=False)
        ])
    elif args.test_crops == 10: # 5 normal crops + 5 flipped crops
        cropping = torchvision.transforms.Compose([
            GroupOverSample(
                crop_size=input_size,
                scale_size=scale_size,
            )
        ])
    else:
        raise ValueError("Only 1, 3, 5, 10 crops are supported while we got {}".format(args.test_crops))


    val_data = Video_dataset(
        config.data.val_root, config.data.val_list, config.data.label_list,
        random_shift=False, num_segments=config.data.num_segments,
        modality=config.data.modality,
        image_tmpl=config.data.image_tmpl,
        test_mode=True,
        transform=torchvision.transforms.Compose([
            cropping,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            GroupNormalize(input_mean, input_std),
        ]),
        dense_sample=args.dense,
        test_clips=args.test_clips)

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
    val_loader = DataLoader(val_data,
        batch_size=config.data.batch_size, num_workers=config.data.workers,
        sampler=val_sampler, pin_memory=True, drop_last=False)


    # # ============= generate class features ==============
    # logger.info('============= Start encoding class features ===========')
    # classes = text_prompt_ensemble(val_data)
    # n_class = classes[0].size(0)
    # model_clip.cuda()
    # model_clip.eval()
    # with torch.no_grad():
    #     # @zmhh_h multi text prompts
    #     cls_feature_list = [model_clip.encode_text(classes[i].cuda(), return_token=True)[0] for i in range(len(classes))]
    #     for cls_feature in cls_feature_list:
    #         cls_feature /= cls_feature.norm(dim=-1, keepdim=True)
    #     cls_feature = torch.stack(cls_feature_list, 0).mean(0)
    #     cls_feature /= cls_feature.norm(dim=-1, keepdim=True)
    # logger.info('============= End encoding class features ===========')

    # tokenized specific text prompts
    # classes: dict:(num_template : [num_cls, 77])
    classes = text_prompt_multi(val_data)
    # n_class = classes[0].size(0)

    model = VideoCLIP(model_clip, config.data.num_segments)
    del model_clip

    # Temporal Aggregation Module
    video_head = video_header(
        config.network.sim_header,
        config.network.interaction,
        clip_state_dict,
        config.network.temporal_layer)

    # ===================== normally load weights ===============================
    if os.path.isfile(args.weights):
        logger.info("=> loading checkpoint '{}'".format(args.weights))
        checkpoint = torch.load(args.weights, map_location='cpu')
        if dist.get_rank() == 0:
            logger.info('load model: epoch {}'.format(checkpoint['epoch']))

        model.load_state_dict(update_dict(checkpoint['model_state_dict']))
        video_head.load_state_dict(update_dict(checkpoint['fusion_model_state_dict']))
        del checkpoint
    else:
        logger.info("=> no checkpoint found at '{}'".format(args.weights))
    # =============== patch clip weights with a ratio of alphe===================
    # if os.path.isfile(args.weights):
    #     logger.info("=> loading checkpoint '{}'".format(args.weights))
    #     checkpoint = torch.load(args.weights, map_location='cpu')
    #     print('keys ',checkpoint['model_state_dict'].keys())
    #     checkpoint_patch = {}
    #     alpha = 1.0
    #     for k, v in checkpoint['model_state_dict'].items():
    #         if k in clip_state_dict.keys(): # logit scale?
    #             checkpoint_patch[k]= alpha*v+(1-alpha)*clip_state_dict[k]
    #         else:
    #             logger.info('unmatched parameters: ',k)
    #     if dist.get_rank() == 0:
    #         logger.info('load model: epoch {}'.format(checkpoint['epoch']))
    #     # model.load_state_dict(update_dict(checkpoint['model_state_dict']))
    #     model.load_state_dict(checkpoint_patch)
    #     video_head.load_state_dict(update_dict(checkpoint['fusion_model_state_dict']))
    #     del checkpoint,checkpoint_patch
    # else:
    #     logger.info("=> no checkpoint found at '{}'".format(args.weights))

    if args.distributed:
        model = DistributedDataParallel(model.cuda(), device_ids=[args.gpu], find_unused_parameters=True)
        if config.network.sim_header not in ["None", "Selective"]:
            video_head = DistributedDataParallel(video_head.cuda(), device_ids=[args.gpu])


    prec1 = validate(
        val_loader, device,
        model, video_head, config, classes, args.test_crops, args.test_clips, logger)
    return


def validate(val_loader, device, model, video_head, config, classes, test_crops, test_clips, logger):

    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    video_head.eval()
    proc_start_time = time.time()

    sim_logits = []     # 
    labels = []     # 
    i_features = []

    with torch.no_grad():
        logger.info('============= Start encoding class features ===========')
        text = classes
        text_embedding_list = [model.module.encode_text(text[i].cuda(), return_token=False)[0] for i in range(len(text))]
        # mean(0) first of last?? 
        text_embedding = torch.stack(text_embedding_list, 0).mean(0) # [num_cls, C]
        text_embedding /= text_embedding.norm(dim=-1, keepdim=True)  # [num_cls, C]
        n_class = text_embedding.size(0)
        logger.info('============= End encoding class features ===========')

        for i, (image, class_id) in enumerate(val_loader):
            batch_size = class_id.numel()
            num_crop = test_crops
            num_crop *= test_clips  # 4 clips for testing when using dense sample

            class_id = class_id.to(device) # [BS]
            n_seg = config.data.num_segments
            image = image.view((-1, n_seg, 3) + image.size()[-2:]) # [BS*num_crop, T, 3, H ,W]
            b, t, c, h, w = image.size()
            # # ============ visualize frames ==========
            # import utils.visualization
            # for i in range(image.shape[0]):
            #     utils.visualization.show_video_frame(image[i].permute(1,0,2,3))
            # # ============ end visualize =============
            image_input = image.to(device).view(-1, c, h, w) # [BS*num_crop*T, 3, H, W]
            image_features = model.module.encode_image(image_input) # [BS*num_crop, T, C]
            image_features_multiview = image_features.reshape(batch_size,num_crop,t,-1).mean(2) # @zmhh_h [BS, num_crop, C]
            
            cnt_time = time.time() - proc_start_time
            similarity = video_head(image_features, text_embedding)  # [BS*num_crop, num_class]
            similarity = F.softmax(similarity, -1) # 这时候是每一个clip对应各class的概率
            similarity = similarity.reshape(batch_size, num_crop, -1).mean(1)  # [BS, num_class] 多clip相似度融合后所有概率和可能不为1
            similarity = similarity.view(batch_size, -1, n_class).softmax(dim=-1) # [BS, 1, num_class] 再经过softmax
            similarity = similarity.mean(dim=1, keepdim=False)  # [BS, num_class]

            ########## gathering 
            # i_features.append(concat_all_gather(image_features))
            i_features.append(concat_all_gather(image_features_multiview)) # @zmhh_h
            sim_logits.append(concat_all_gather(similarity))
            labels.append(concat_all_gather(class_id))
            ##########


            prec = accuracy(similarity, class_id, topk=(1, 5))
            prec1 = reduce_tensor(prec[0])
            prec5 = reduce_tensor(prec[1])

            top1.update(prec1.item(), class_id.size(0))
            top5.update(prec5.item(), class_id.size(0))

            if i % config.logging.print_freq == 0 and dist.get_rank() == 0:
                runtime = float(cnt_time) / (i+1) / (batch_size * dist.get_world_size())
                logger.info(
                    ('Test: [{0}/{1}], average {runtime:.4f} sec/video \t'
                     'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                     'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), runtime=runtime, top1=top1, top5=top5)))
    logger.info(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5)))

    if dist.get_rank() == 0:
        ## half-classes evaluation
        sim, la = sim_logits[0], labels[0]
        vid_feat = i_features[0]
        for i in range(1, len(sim_logits)): 
            sim = torch.cat((sim, sim_logits[i]), 0)
            la = torch.cat((la, labels[i]), 0)
            vid_feat = torch.cat((vid_feat, i_features[i]), 0)

        text_emb = text_embedding/ text_embedding.norm(dim=-1, keepdim=True)
        vid_feat = vid_feat/ vid_feat.norm(dim=-1, keepdim=True) # @zmhh_h

        # torch.save(text_emb, 'S14_9_TEXT_HMDB.pt')
        # torch.save(vid_feat, 'S14_9_VISUAL_HMDB.pt')

        # ============ zero shot with features directly from CLIP ======================
        logger.info('============================================================')
        full_acc1, full_acc5, acc_split, acc_split_top5 = multi_split_test(vid_feat.cpu(), text_emb.cpu(), la.cpu())
        logger.info('---------- Full-classes Evaluation CLIP ----------')
        logger.info('Overall Top1 {:.03f}% Top5 {:.03f}%'.format(full_acc1.item(), full_acc5.item()))
        accuracy_split, accuracy_split_std = np.mean(acc_split), np.std(acc_split)
        accuracy_split_top5, accuracy_split_top5_std = np.mean(acc_split_top5), np.std(acc_split_top5)
        logger.info('---------- Half-classes Evaluation CLIP ----------')
        logger.info('Top1: mean {:.03f}%, std {:.03f}%'.format(accuracy_split, accuracy_split_std))
        logger.info('Top5: mean {:.03f}%, std {:.03f}%'.format(accuracy_split_top5, accuracy_split_top5_std))  

        logger.info('============================================================')
    return top1.avg


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output.cpu()

def compute_accuracy(vis_emb, text_emb, label):
    n_class = len(text_emb) # int: num_class
    n_samples = len(vis_emb) # int: num_video
    similarity=(100.0 * vis_emb @ text_emb.T) # [num_video, num_crop, num_class]
    # print('similarity',similarity.shape)
    # print('similarityyy',similarity[0,0,:])

    # # normalize then fuse
    # similarity=similarity.view(n_samples, -1, n_class).softmax(dim = -1)
    # similarity=similarity.mean(dim = 1, keepdim = False)  # b 101 [num_videos, num_classes]

    # fuse then normalize
    similarity=similarity.mean(dim = 1, keepdim = False)  # b 101 [num_videos, num_classes]
    similarity=similarity.view(n_samples, n_class).softmax(dim = -1)
    
    # similarity: [num_videos, num_classes]
    prec=accuracy(similarity, label, topk = (1, 5))
    return prec[0], prec[1]
 
 
def multi_split_test(vis_embs, text_embs, true_label):
    # vis_embs: [10000, 768] [num_video, num_crop, C]
    # text_embs: [101, 768] [num_class, C]
    # true_label: [10000,] [num_video]
    full_acc1, full_acc5 = compute_accuracy(vis_embs, text_embs, true_label)
    # print('-----Full-classes Evaluation------')
    # print('Overall Top1 {:.03f}% Top5 {:.03f}%'.format(full_acc1.item(), full_acc5.item()))
 
    # Calculate accuracy per split
    # Only when the model has been trained on a different dataset
    true_label = true_label.numpy()
    accuracy_split, accuracy_split_top5 = np.zeros(10), np.zeros(10)
    for split in range(len(accuracy_split)):
        np.random.seed(split)
        sel_classes = np.random.permutation(len(text_embs))[:len(text_embs) // 2]  # [50, ]
        sel = [l in sel_classes for l in true_label]    # len = 10000 [<num_video]
        subclasses = np.unique(true_label[sel])         # [num_class//2 ]
        # label_map = {}
        # for i in range(len(subclasses)):
        #     label_map[subclasses[i]] = i
        # new_label = np.array([label_map[l] for l in true_label[sel]])
        # new_label = torch.from_numpy(new_label)
        # label mapping: [4900, ], new label
        tl = np.array([int(np.where(l == subclasses)[0]) for l in true_label[sel]])
        tl = torch.from_numpy(tl)
        acc, acc5 = compute_accuracy(vis_embs[sel], text_embs[subclasses], tl)
        accuracy_split[split] = acc
        accuracy_split_top5[split] = acc5
    
    return full_acc1, full_acc5, accuracy_split, accuracy_split_top5


if __name__ == '__main__':
    args = get_parser()
    main(args)
