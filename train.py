import comet_ml
from datetime import datetime

import torch
import torch.multiprocessing as mp
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
import random

torch.manual_seed(1)  # cpu
torch.cuda.manual_seed_all(1)  # gpu
np.random.seed(1)  # numpy
random.seed(1)  # random and transforms
torch.backends.cudnn.deterministic = True  # cudnn
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import time

from config import config_dict
from datasets.generateData import generate_dataset
from net.generateNet import generate_net
import torch.optim as optim
from PIL import Image
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from net.sync_batchnorm.replicate import patch_replication_callback
from utils.configuration import Configuration
from utils.finalprocess import writelog
from utils.imutils import img_denorm
from net.sync_batchnorm import SynchronizedBatchNorm2d
from utils.visualization import generate_vis, max_norm
from tqdm import tqdm
import argparse
from utils.JSD_loss import calc_jsd_multiscale as calc_jsd_temp
from utils.eval_net_utils import eval_net_multiprocess
from scipy.optimize import curve_fit
import json
from collections import OrderedDict
import pickle
from utils.iou_computation import update_iou_stat, compute_iou, iter_iou_stat, get_mask, iter_fraction_pixelwise
from utils.logger import CometWriter

cfg = Configuration(config_dict)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class worker_init_fn:
    def __init__(self, worker_id):
        self.id = worker_id

    def __call__(self):
        return np.random.seed(1 + self.id)


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--EXP_NAME", type=str, default=cfg.EXP_NAME,
                        help="the name of the experiment")
    parser.add_argument("--scale_factor", type=float, default=cfg.scale_factor,
                        help="scale_factor of downsample the image")
    parser.add_argument("--scale_factor2", type=float, default=cfg.scale_factor2,
                        help="scale_factor of upsample the image")
    parser.add_argument("--DATA_PSEUDO_GT", type=str, default=cfg.DATA_PSEUDO_GT,
                        help="Data path for the main segmentation map")
    parser.add_argument("--TRAIN_CKPT", type=str, default=cfg.TRAIN_CKPT,
                        help="Training path")
    parser.add_argument("--Lambda1", type=float, default=1,
                        help="to balance the loss between CE and Consistency loss")
    parser.add_argument("--TRAIN_BATCHES", type=int, default=cfg.TRAIN_BATCHES,
                        help="training batch szie")
    parser.add_argument('--threshold', type=float, default=0.8,
                        help="threshold to select the mask, ")
    parser.add_argument('--DATA_WORKERS', type=int, default=cfg.DATA_WORKERS,
                        help="number of workers in dataloader")

    parser.add_argument('--TRAIN_LR', type=float,
                        default=cfg.TRAIN_LR,
                        help="the path of trained weight")
    parser.add_argument('--TRAIN_ITERATION', type=int,
                        default=cfg.TRAIN_ITERATION,
                        help="the training iteration number")
    parser.add_argument('--DATA_RANDOMCROP', type=int, default=cfg.DATA_RANDOMCROP,
                        help="the resolution of random crop")



    # related to the pseudo label updating
    parser.add_argument('--mask_threshold', type=float, default=0.8,
                        help="only the region with high probability and disagree with Pseudo label be updated")
    parser.add_argument('--update_interval', type=int, default=1,
                        help="evaluate the prediction every 1 epoch")
    parser.add_argument('--npl_metrics', type=int, default=0,
                        help="0: using the original cam to compute the npl similarity, 1: use the updated pseudo label to compute the npl")
    parser.add_argument('--r_threshold', type=float, default=0.9,
                        help="the r threshold to decide if_update")

    # related to the eval mode
    parser.add_argument('--scale_index', type=int, default=2,
                        help="0: scale [0.7, 1.0, 1.5]  1:[0.5, 1.0, 1.75], 2:[0.5, 0.75, 1.0, 1.25, 1.5, 1.75] ")
    parser.add_argument('--flip', type=str, default='yes',
                        help="do not flip in the eval pred if no, else flip")
    parser.add_argument('--CRF', type=str, default='no',
                        help="whether to use CRF, yes or no, default no")
    parser.add_argument('--dict_save_scale_factor', type=float, default=1,
                        help="dict_save_scale_factor downsample_factor (in case the CPU memory is not enough)")
    parser.add_argument('--evaluate_interval', type=int, default=1,
                        help="evaluate the prediction every 1 epoch, this is always set to one for PASCAL VOC dataset")
    parser.add_argument('--Reinit_dict', type=str2bool, nargs='?',
                        const=True, default=False,
                        help="whether to reinit the dict every epoch")
    parser.add_argument('--evaluate_aug_epoch', type=int, default=9,
                        help="when to start aug the evaluate with CRF and flip, this can be used to save some time when updating the pseudo label, we did not find significant difference")



    # continue_training_related:
    parser.add_argument('--continue_train_epoch', type=int, default=0,
                        help="load the trained model from which epoch, if 0, no continue training")
    parser.add_argument('--checkpoint_path', type=str, default='no',
                        help="the checkpoint path to load the model")
    parser.add_argument('--dict_path', type=str,
                        default='no',
                        help="the dict path of seg path")
    parser.add_argument('--MODEL_BACKBONE_PRETRAIN', type=str2bool, nargs='?',
                        const=True, default=True,
                        help="Do not load pretrained model if false")


    # Comet
    parser.add_argument('--api_key', type=str,
                        default='',
                        help="The api_key of Comet")
    parser.add_argument('--online', type=str2bool, nargs='?',
                        const=True, default=True,
                        help="False when use Comet offline")

    return parser.parse_args()


def curve_func(x, a, b, c):
    return a * (1 - np.exp(-1 / c * x ** b))


def fit(func, x, y):
    popt, pcov = curve_fit(func, x, y, p0=(1, 1, 1), method='trf', sigma=np.geomspace(1, .1, len(y)),
                           absolute_sigma=True, bounds=([0, 0, 0], [1, 1, np.inf]))
    return tuple(popt)


def derivation(x, a, b, c):
    x = x + 1e-6  # numerical robustness
    return a * b * 1 / c * np.exp(-1 / c * x ** b) * (x ** (b - 1))


def label_update_epoch(ydata_fit, threshold=0.9, eval_interval=100, num_iter_per_epoch=10581 / 10):
    xdata_fit = np.linspace(0, len(ydata_fit) * eval_interval / num_iter_per_epoch, len(ydata_fit))
    a, b, c = fit(curve_func, xdata_fit, ydata_fit)
    epoch = np.arange(1, 16)
    y_hat = curve_func(epoch, a, b, c)
    relative_change = abs(abs(derivation(epoch, a, b, c)) - abs(derivation(1, a, b, c))) / abs(derivation(1, a, b, c))
    relative_change[relative_change > 1] = 0
    update_epoch = np.sum(relative_change <= threshold) + 1
    return update_epoch  # , a, b, c


def if_update(iou_value, current_epoch, threshold=0.90):
    update_epoch = label_update_epoch(iou_value, threshold=threshold)
    return current_epoch >= update_epoch  # , update_epoch

def train_net():
    args = get_arguments()

    cfg.MODEL_SAVE_DIR = os.path.join(cfg.ROOT_DIR, 'model', args.EXP_NAME)
    cfg.LOG_DIR = os.path.join(cfg.ROOT_DIR, 'log', args.EXP_NAME)
    cfg.DATA_PSEUDO_GT = args.DATA_PSEUDO_GT
    cfg.DATA_NAME = 'VOCTrainwsegDataset'
    cfg.TRAIN_LR = args.TRAIN_LR
    cfg.MODEL_NAME = 'deeplabv1_wo_interp'
    cfg.DATA_RANDOMCROP = args.DATA_RANDOMCROP
    cfg.MODEL_BACKBONE_PRETRAIN = args.MODEL_BACKBONE_PRETRAIN

    if not os.path.exists(cfg.MODEL_SAVE_DIR):
        os.mkdir(cfg.MODEL_SAVE_DIR)
    if not os.path.exists(cfg.LOG_DIR):
        os.mkdir(cfg.LOG_DIR)

    # save args setting in the log file
    with open(os.path.join(cfg.LOG_DIR, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # initialize writer
    writer = CometWriter(
        project_name="noisy-segmentation",
        experiment_name=args.EXP_NAME + '-' + datetime.now().strftime("%m:%d:%H:%M"),
        api_key=args.api_key,
        log_dir=cfg.LOG_DIR,
        offline=((not args.online) and (args.api_key == '')))

    period = 'train'
    transform = 'weak'



    cfg_eval = Configuration(cfg.__dict__.copy())
    # do not want to save the segmentation in the seg_dict for the eval dataset
    cfg_eval.DATA_NAME = 'VOCEvalDataset'
    dataset = generate_dataset(cfg, period=period, transform=transform)

    scale_index = args.scale_index
    if scale_index == 0:
        TEST_MULTISCALE = [0.75, 1.0, 1.5]
    elif scale_index == 1:
        TEST_MULTISCALE = [0.5, 1.0, 1.75]
    elif scale_index == 2:
        TEST_MULTISCALE = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    elif scale_index == 3:
        TEST_MULTISCALE = [0.7, 1.0, 1.5]
    elif scale_index == 4:
        TEST_MULTISCALE = [0.5, 0.75, 1.0, 1.25, 1.5]
    elif scale_index == 5:
        TEST_MULTISCALE = [1]

    cfg_eval.TEST_MULTISCALE = TEST_MULTISCALE
    # cfg_eval.TEST_MULTISCALE = [0.5, 0.7, 0.75, 1.0, 1.25, 1.5, 1.75]


    # bs = 1, one by one eval
    evalset = generate_dataset(cfg_eval, period=period, transform='none')
    evalset2 = generate_dataset(cfg_eval, period=period, transform='none')


    indxset_shuffle = np.arange(len(dataset.name_list))
    np.random.shuffle(indxset_shuffle)

    evalset.ori_indx_list = indxset_shuffle[0:int(0.5 * (len(dataset.name_list)))]
    evalset.name_list = evalset.name_list[evalset.ori_indx_list]

    evalset2.ori_indx_list = indxset_shuffle[int(0.5 * (len(dataset.name_list))):len(dataset.name_list)]
    evalset2.name_list = evalset2.name_list[evalset2.ori_indx_list]


    dataloader = DataLoader(dataset,
                            batch_size=args.TRAIN_BATCHES,
                            shuffle=cfg.TRAIN_SHUFFLE,
                            num_workers=args.DATA_WORKERS,
                            multiprocessing_context=SpawnContext,
                            pin_memory=True,
                            drop_last=True,
                            worker_init_fn=worker_init_fn)
    # if one by one eval, the batch size is set to 1
    eval_dataloader1 = DataLoader(evalset,
                                  batch_size=1,
                                  shuffle=False,
                                  pin_memory=True,
                                  drop_last=False)
    eval_dataloader2 = DataLoader(evalset2,
                                  batch_size=1,
                                  shuffle=False,
                                  pin_memory=True,
                                  drop_last=False)
    # load the previous checkpoint
    if args.checkpoint_path != 'no':
        checkpoint = torch.load(args.checkpoint_path)

    net = generate_net(cfg, batchnorm=nn.BatchNorm2d)
    # load the model
    if args.checkpoint_path != 'no':
        net.load_state_dict(checkpoint['net'])

    if cfg.TRAIN_TBLOG:
        from tensorboardX import SummaryWriter
        # Set the Tensorboard logger
        tblogger = SummaryWriter(cfg.LOG_DIR)

    print('Use %d GPU' % cfg.GPUS)
    device = torch.device(0)
    if cfg.GPUS > 1:
        net = nn.DataParallel(net)
        patch_replication_callback(net)
        parameter_source = net.module

    else:
        parameter_source = net
    net.to(device)


    # those two are used for eval only, no gradient is needed, for the multiprocess evaluation purpose
    net1 = generate_net(cfg, batchnorm=nn.BatchNorm2d)
    net2 = generate_net(cfg, batchnorm=nn.BatchNorm2d)
    for param in net1.parameters():
        param.detach_()
    for param in net2.parameters():
        param.detach_()

    net_state_dict = net.state_dict()
    new_state_dict = OrderedDict()
    for k, v in net_state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    net1.load_state_dict(new_state_dict, strict=True)
    net1.to(torch.device(0))
    net2.load_state_dict(new_state_dict, strict=True)
    net2.to(torch.device(1))
    del new_state_dict

    weight = nn.Parameter(torch.Tensor(3))
    weight.data.fill_(1)
    weight.to(device)

    parameter_groups = parameter_source.get_parameter_groups()


    optimizer = optim.SGD(
        params=[
            {'params': parameter_groups[0], 'lr': args.TRAIN_LR, 'weight_decay': cfg.TRAIN_WEIGHT_DECAY},
            {'params': parameter_groups[1], 'lr': 2 * args.TRAIN_LR, 'weight_decay': 0},
            {'params': parameter_groups[2], 'lr': 10 * args.TRAIN_LR, 'weight_decay': cfg.TRAIN_WEIGHT_DECAY},
            {'params': parameter_groups[3], 'lr': 20 * args.TRAIN_LR, 'weight_decay': 0},
            {'params': weight, 'lr': args.TRAIN_LR, 'weight_decay': 0}
        ],
        momentum=cfg.TRAIN_MOMENTUM,
        weight_decay=cfg.TRAIN_WEIGHT_DECAY
    )

    # load the eval history for tb log
    if args.checkpoint_path != 'no':
        # Load IoU curve for contune training
        if 'IoU_dict' in checkpoint.keys():
            IoU_npl_dict = checkpoint['IoU_dict']
            Updated_class_list = checkpoint['updated_class']
        else:
            # use to record the updated class, so that it won't be updated again
            Updated_class_list = []
            # record the noisy pseudo label fitting IoU for each class
            IoU_npl_dict = {}
            for i in range(21):
                IoU_npl_dict[i] = []

    else:
        # use to record the updated class, so that it won't be updated again
        Updated_class_list = []
        # record the noisy pseudo label fitting IoU for each class
        IoU_npl_dict = {}
        for i in range(21):
            IoU_npl_dict[i] = []

    itr = args.continue_train_epoch * len(dataset) // (cfg.TRAIN_BATCHES)
    max_itr = args.TRAIN_ITERATION
    max_epoch = max_itr * (cfg.TRAIN_BATCHES) // len(dataset) + 1
    tblogger = SummaryWriter(cfg.LOG_DIR)

    if len(Updated_class_list) != 0:
        class_names_list = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
                            'chair',
                            'cow',
                            'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                            'train', 'tvmonitor']
        Updated_class_name_list = []
        for class_indx in Updated_class_list:
            Updated_class_name_list.append(class_names_list[class_indx])
        writer.add_text('Previous_updated_class_list' + str(Updated_class_name_list), 0)

    #  load the weight and optimizer
    if args.checkpoint_path != 'no':
        weight = checkpoint['w'].to(device)
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('load previous checkpoint')

    if not args.dict_path == 'no':
        if args.dict_path.endswith('npy'):
            tempt = np.load(args.dict_path, allow_pickle=True)
            dataloader.dataset.seg_dict = tempt[()]
        elif args.dict_path.endswith('pkl'):
            dataloader.dataset.seg_dict = pickle.load(open(args.dict_path, "rb"))
    else:
        # if not train from scratch, no previous dict to load, reevaluate it
        if args.continue_train_epoch != 0:
            IoU_npl_indx = np.array([0] + Updated_class_list)

            eval_net_multiprocess(SpawnContext, net1, net2, IoU_npl_indx, dataloader, eval_dataloader1,
                                      eval_dataloader2,
                                      momentum=0, scale_index=args.scale_index, flip=args.flip,
                                      scalefactor=args.dict_save_scale_factor, CRF_post=args.CRF,
                                      tempt_save_root=cfg.LOG_DIR)
            print('pred_done!')
            # update the segmentation label
            dataloader.dataset.update_seg_dict(IoU_npl_indx, mask_threshold=args.mask_threshold)
            # clean the prev_pred_dict to save CPU memory
            dataloader.dataset.prev_pred_dict.clear()
            np.save(os.path.join(cfg.MODEL_SAVE_DIR, 'seg_dict.npy'), dataloader.dataset.seg_dict)

    # eval the pseudo label quality
    loglist = dataloader.dataset.do_python_eval_batch_pseudo_one_process()
    for indx, class_name in enumerate(
            ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
             'cow',
             'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
             'tvmonitor', 'mIoU']):
        writer.add_scalar({'pseudolabel_dict_' + class_name: loglist[class_name]}, step=itr)
        # experiment.log_metric('pseudolabel_dict_' + class_name, loglist[class_name], step=itr)


    writer.log_hyperparams(args)
    writer.log_code(folder='./lib/utils/')
    writer.log_code(file_name='./train.py')
    writer.log_code(file_name=None, folder='./lib/datasets')



    with tqdm(total=max_itr) as pbar:
        for epoch in range(args.continue_train_epoch, max_epoch):
            TP_clean = [0] * 21
            P_clean = [0] * 21
            T_clean = [0] * 21

            TP_wrong = [0] * 21
            P_wrong = [0] * 21
            T_wrong = [0] * 21

            # noisy pseudo label fit
            TP_npl = [0] * 21
            P_npl = [0] * 21
            T_npl = [0] * 21

            # stat for each epoch
            TP_clean_epoch = [0] * 21
            P_clean_epoch = [0] * 21
            T_clean_epoch = [0] * 21

            TP_wrong_epoch = [0] * 21
            P_wrong_epoch = [0] * 21
            T_wrong_epoch = [0] * 21

            # record all the statistics
            TP_gt_epoch = [0] * 21
            P_gt_epoch = [0] * 21
            T_gt_epoch = [0] * 21

            # the updated pseudo label quality
            TP_pl_epoch = [0] * 21
            P_pl_epoch = [0] * 21
            T_pl_epoch = [0] * 21

            # the noisy label fit IoU
            TP_npl_epoch = [0] * 21
            P_npl_epoch = [0] * 21
            T_npl_epoch = [0] * 21

            for i_batch, sample in enumerate(dataloader):

                now_lr = adjust_lr(optimizer, itr, max_itr, args.TRAIN_LR, cfg.TRAIN_POWER)
                optimizer.zero_grad()

                inputs, seg_label, seg_GT = sample['image'], sample['segmentation'], sample['segmentationgt']
                seg_ori_ST = sample['segmentation2'].clone()

                n, c, h, w = inputs.size()


                inputs_small = F.interpolate(inputs, scale_factor=args.scale_factor, mode='bilinear',
                                             align_corners=True,
                                             recompute_scale_factor=True)

                inputs_large = F.interpolate(inputs, scale_factor=args.scale_factor2, mode='bilinear',
                                             align_corners=True,
                                             recompute_scale_factor=True)

                pred1 = net(inputs.to(device))
                pred1 = F.interpolate(pred1, size=(h, w), mode='bilinear', align_corners=True)


                # input to be scaled e.g 0.7
                pred2 = net(inputs_small.to(device))
                pred2 = F.interpolate(pred2, size=(h, w), mode='bilinear', align_corners=True)

                # input to be scaled e.g 1.5
                pred3 = net(inputs_large.to(device))
                pred3 = F.interpolate(pred3, size=(h, w), mode='bilinear', align_corners=True)

                pred_np = torch.argmax(pred1, dim=1).detach().cpu().numpy()  # b, h, w
                gt_np = seg_GT.detach().cpu().numpy()
                # label_np = seg_label.numpy()


                label_np = seg_ori_ST.numpy()
                if args.npl_metrics == 0:
                    label_np_updated = seg_ori_ST.numpy()
                else:
                    label_np_updated = seg_label.numpy()

                # the visualization of the label memorization
                mask_clean = (gt_np == label_np)

                gt_np_clean = (gt_np + 1) * mask_clean - 1
                gt_np_clean[gt_np_clean < 0] = 255

                gt_np_wrong = (gt_np + 1) * (~mask_clean) - 1
                gt_np_wrong[gt_np_wrong < 0] = 255

                label_np_clean = (label_np + 1) * mask_clean - 1
                label_np_clean[label_np_clean < 0] = 255

                label_np_wrong = (label_np + 1) * (~mask_clean) - 1
                label_np_wrong[label_np_wrong < 0] = 255

                TP_clean, P_clean, T_clean = update_iou_stat(pred_np, gt_np_clean, TP_clean, P_clean, T_clean)
                TP_clean_epoch, P_clean_epoch, T_clean_epoch = update_iou_stat(pred_np, gt_np_clean, TP_clean_epoch,
                                                                               P_clean_epoch, T_clean_epoch)

                TP_wrong, P_wrong, T_wrong = update_iou_stat(pred_np, gt_np_wrong, TP_wrong, P_wrong, T_wrong)
                TP_wrong_epoch, P_wrong_epoch, T_wrong_epoch = update_iou_stat(pred_np, gt_np_wrong, TP_wrong_epoch,
                                                                               P_wrong_epoch, T_wrong_epoch)

                TP_gt_epoch, P_gt_epoch, T_gt_epoch = update_iou_stat(pred_np, gt_np, TP_gt_epoch,
                                                                      P_gt_epoch, T_gt_epoch)

                TP_pl_epoch, P_pl_epoch, T_pl_epoch = update_iou_stat(seg_label.detach().cpu().numpy(), gt_np,
                                                                      TP_pl_epoch,
                                                                      P_pl_epoch, T_pl_epoch)

                # the statistics about noise segmentation label fitting
                TP_npl, P_npl, T_npl = update_iou_stat(pred_np, label_np_updated, TP_npl,
                                                       P_npl, T_npl)

                TP_npl_epoch, P_npl_epoch, T_npl_epoch = update_iou_stat(pred_np, label_np_updated, TP_npl_epoch,
                                                                         P_npl_epoch, T_npl_epoch)


                # CE loss and the consistency loss
                loss_ce, consistency, variance, mixture_label = calc_jsd_temp(
                    weight.to(device), seg_label.to(device), pred1,
                    pred2, pred3, threshold=args.threshold)
                loss = loss_ce + args.Lambda1 * consistency


                # check the loss w.r.t GT, this is only used for visualization and analysis, not used for training
                criterion_GT = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
                loss_clean = criterion_GT(pred1, torch.tensor(label_np_clean).to(0))
                loss_wrong = criterion_GT(pred1, torch.tensor(label_np_wrong).to(0))

                loss.backward()
                optimizer.step()

                pbar.set_description("loss=%g " % (loss.item()))
                pbar.update(1)

                time.sleep(0.001)
                # for visualization and training metircs
                if cfg.TRAIN_TBLOG and itr % 100 == 0:
                    if  int(sample['batch_idx'][-1].cpu().numpy()) in dataloader.dataset.prev_pred_dict.keys():
                        prev_predict_vis = torch.tensor(dataloader.dataset.prev_pred_dict[
                                                            int(sample['batch_idx'][-1].cpu().numpy())])  # 1,c,h/4,w/4
                        b, c, h, w = prev_predict_vis.size()
                        mask_seg_prednan_vis = torch.isnan(
                            prev_predict_vis)  # the place where the value is nan in the maskprediction  b,c,h,w
                        seg_argmax_vis = torch.ones((b, h, w), dtype=torch.long) * 255
                        seg_argmax_vis[~mask_seg_prednan_vis[:, 0, :, :]] = torch.argmax(prev_predict_vis, dim=1)[
                            ~mask_seg_prednan_vis[:, 0, :, :]]  # b,h,w
                        seg_argmax_vis_color = dataset.label2colormap(seg_argmax_vis[0].cpu().numpy()).transpose(
                            (2, 0, 1))
                        tblogger.add_image('seg_argmax_vis_dict', seg_argmax_vis_color, itr)

                    inputs1 = img_denorm(inputs[-1].cpu().numpy()).astype(np.uint8)
                    label1 = seg_ori_ST[-1].cpu().numpy()
                    # label1 = sample['segmentation'][-1].cpu().numpy()
                    label_color1 = dataset.label2colormap(label1).transpose((2, 0, 1))

                    n, c, h, w = inputs.size()
                    seg_vis1 = torch.argmax(pred1[-1], dim=0).detach().cpu().numpy()
                    seg_color1 = dataset.label2colormap(seg_vis1).transpose((2, 0, 1))


                    seg_vis2 = torch.argmax(pred2[-1], dim=0).detach().cpu().numpy()
                    seg_color2 = dataset.label2colormap(seg_vis2).transpose((2, 0, 1))

                    seg_vis3 = torch.argmax(pred3[-1], dim=0).detach().cpu().numpy()
                    seg_color3 = dataset.label2colormap(seg_vis3).transpose((2, 0, 1))

                    mixture_label_vis = torch.argmax(mixture_label[-1], dim=0).detach().cpu().numpy()
                    mixture_label_color = dataset.label2colormap(mixture_label_vis).transpose((2, 0, 1))

                    label_GT = sample['segmentationgt'][-1].cpu().numpy()
                    label_colorGT = dataset.label2colormap(label_GT).transpose((2, 0, 1))

                    IoU_clean = compute_iou(TP_clean, P_clean, T_clean)
                    IoU_wrong = compute_iou(TP_wrong, P_wrong, T_wrong)
                    IoU_npl = compute_iou(TP_npl, P_npl, T_npl)

                    for i in range(21):
                        IoU_npl_dict[i].append(IoU_npl[i])

                    for indx, class_name in enumerate(
                            ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
                             'chair', 'cow',
                             'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                             'train', 'tvmonitor']):
                        writer.add_scalar({'clean_' + class_name: IoU_clean[indx]}, step=itr)
                        writer.add_scalar({'wrong_' + class_name: IoU_wrong[indx]}, step=itr)
                        writer.add_scalar({'npl_' + class_name: IoU_npl[indx]}, step=itr)

                    mIoU_clean = np.mean(np.array(IoU_clean))
                    mIoU_wrong = np.mean(np.array(IoU_wrong))
                    mIoU_npl = np.mean(np.array(IoU_npl))
                    writer.add_scalar({'mIoU_clean': mIoU_clean}, itr)
                    writer.add_scalar({'mIoU_wrong': mIoU_wrong}, itr)
                    writer.add_scalar({'mIoU_npl': mIoU_npl}, itr)

                    # reset the TP, P, T.
                    TP_clean = [0] * 21
                    P_clean = [0] * 21
                    T_clean = [0] * 21

                    TP_wrong = [0] * 21
                    P_wrong = [0] * 21
                    T_wrong = [0] * 21

                    TP_npl = [0] * 21
                    P_npl = [0] * 21
                    T_npl = [0] * 21

                    writer.add_scalar({'loss': loss.item()}, itr)

                    writer.add_scalar({'lossGT_clean': torch.mean(loss_clean, dim=(0, 1, 2)).item()}, itr)
                    writer.add_scalar({'lossGT_wrong': torch.mean(loss_wrong, dim=(0, 1, 2)).item()}, itr)

                    writer.add_scalar({'loss_ce': loss_ce.item()}, itr)
                    writer.add_scalar({'consistency': consistency.item()}, itr)

                    writer.add_scalar({'lr': now_lr}, itr)
                    tblogger.add_image('Input', inputs1, itr)
                    tblogger.add_image('Label', label_color1, itr)
                    tblogger.add_image('label_GT', label_colorGT, itr)
                    tblogger.add_image('SEG1', seg_color1, itr)

                    tblogger.add_image('SEG2', seg_color2, itr)
                    tblogger.add_image('SEG3', seg_color3, itr)
                    tblogger.add_image('Weighted_SEG', mixture_label_color, itr)

                    tblogger.add_image('variance', variance[-1].data.cpu().numpy(), itr,
                                       dataformats='HW')

                    # vis the weight
                    writer.add_scalar({'weight1': weight[0].detach().cpu().numpy()}, itr)
                    writer.add_scalar({'weight2': weight[1].detach().cpu().numpy()}, itr)
                    writer.add_scalar({'weight3': weight[2].detach().cpu().numpy()}, itr)


                    label_updated = seg_label[-1].cpu().numpy()
                    label_updated_color = dataset.label2colormap(label_updated).transpose((2, 0, 1))
                    tblogger.add_image('Label_updated', label_updated_color, itr)

                itr += 1
                if itr >= max_itr:
                    break

            # decide which class to update in this epoch
            # the background class will always appear in the update list
            # the already updated class will be updated at each epoch afterwards
            IoU_npl_indx = [0] + Updated_class_list

            for class_idx in range(1, 21):
                # current code only support update each class once, if updated, it won't be updated again
                if not class_idx in Updated_class_list:
                    update_sign = if_update(np.array(IoU_npl_dict[class_idx]), epoch, threshold=args.r_threshold)
                    if update_sign:
                        IoU_npl_indx.append(class_idx)
                        Updated_class_list.append(class_idx)

            # the classes that need to be updated in the current epoch
            IoU_npl_indx = np.array(IoU_npl_indx)

            if epoch < args.evaluate_aug_epoch:
                Def_CRF = 'no'
                Def_flip = 'no'
            else:
                Def_CRF = args.CRF
                Def_flip = 'yes'

            # display which class is updated in each epoch in the tensorboard
            class_names_list = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
                                'chair', 'cow',
                                'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                                'train', 'tvmonitor']
            update_class_name_list = []
            for class_indx in IoU_npl_indx:
                update_class_name_list.append(class_names_list[class_indx])

            writer.add_text('UpdateIndex' + str(update_class_name_list), itr)

            # if only the background class is selected, do not update or eval
            if (epoch % args.evaluate_interval == 0 and len(
                    IoU_npl_indx) > 1):

                if args.Reinit_dict:
                    dataloader.dataset.init_seg_dict()
                # at the end of the epoch, update the dict
                # IoU_npl_indx, which class to update

                net_state_dict = net.state_dict()
                new_state_dict = OrderedDict()
                for k, v in net_state_dict.items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                net1.load_state_dict(new_state_dict)
                net1.to(torch.device(0))
                net2.load_state_dict(new_state_dict)
                net2.to(torch.device(1))
                del new_state_dict

                eval_net_multiprocess(SpawnContext, net1, net2, IoU_npl_indx, dataloader, eval_dataloader1,
                                      eval_dataloader2,
                                      momentum=0, scale_index=args.scale_index, flip=Def_flip,
                                      scalefactor=args.dict_save_scale_factor, CRF_post=Def_CRF,
                                      tempt_save_root=cfg.LOG_DIR,t_eval=3)
                print('pred_done!')

            if epoch % args.update_interval == 0 and len(IoU_npl_indx) > 1:
                # update the segmentation label
                dataloader.dataset.update_seg_dict(IoU_npl_indx,
                                                   mask_threshold=args.mask_threshold)
                # clean the prev_pred_dict to save CPU memory
                dataloader.dataset.prev_pred_dict.clear()

            # let's check the pseudo label performance
            loglist = dataloader.dataset.do_python_eval_batch_pseudo_one_process()
            for indx, class_name in enumerate(
                    ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow',
                     'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                     'tvmonitor', 'mIoU']):
                writer.add_scalar({'pseudolabel_dict_' + class_name: loglist[class_name]}, itr)

            saving_state = {
                'net': parameter_source.state_dict(),
                'w': weight.data.cpu(),
                'optimizer': optimizer.state_dict(),
                'IoU_dict': IoU_npl_dict,
                'updated_class': Updated_class_list
            }

            save_path = os.path.join(cfg.MODEL_SAVE_DIR, '%s_%s_%s_checkpoint-epoch%d.pth' % (
                cfg.MODEL_NAME, cfg.MODEL_BACKBONE, cfg.DATA_NAME, epoch))

            torch.save(saving_state, save_path)

            print('%s has been saved' % save_path)

            IoU_clean_epoch = compute_iou(TP_clean_epoch, P_clean_epoch, T_clean_epoch)
            IoU_wrong_epoch = compute_iou(TP_wrong_epoch, P_wrong_epoch, T_wrong_epoch)
            IoU_gt_epoch = compute_iou(TP_gt_epoch, P_gt_epoch, T_gt_epoch)

            IoU_pl_epoch = compute_iou(TP_pl_epoch, P_pl_epoch, T_pl_epoch)

            IoU_npl_epoch = compute_iou(TP_npl_epoch, P_npl_epoch, T_npl_epoch)

            for indx, class_name in enumerate(
                    ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow',
                     'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                     'tvmonitor']):
                writer.add_scalar({'epoch_clean_' + class_name: IoU_clean_epoch[indx]}, itr)
                writer.add_scalar({'epoch_wrong_' + class_name: IoU_wrong_epoch[indx]}, itr)

                writer.add_scalar({'epoch_gt_' + class_name: IoU_gt_epoch[indx]}, itr)
                writer.add_scalar({'epoch_pl_' + class_name: IoU_pl_epoch[indx]}, itr)
                writer.add_scalar({'epoch_npl_' + class_name: IoU_npl_epoch[indx]}, itr)

            mIoU_clean_epoch = np.mean(np.array(IoU_clean_epoch))
            mIoU_wrong_epoch = np.mean(np.array(IoU_wrong_epoch))
            mIoU_gt_epoch = np.mean(np.array(IoU_gt_epoch))
            mIoU_pl_epoch = np.mean(np.array(IoU_pl_epoch))
            mIoU_npl_epoch = np.mean(np.array(IoU_npl_epoch))

            #
            writer.add_scalar({'epoch_clean_mIoU': mIoU_clean_epoch}, itr)
            writer.add_scalar({'epoch_wrong_mIoU': mIoU_wrong_epoch}, itr)
            writer.add_scalar({'epoch_gt_mIoU': mIoU_gt_epoch}, itr)
            writer.add_scalar({'epoch_pl_mIoU': mIoU_pl_epoch}, itr)
            writer.add_scalar({'epoch_npl_mIoU': mIoU_npl_epoch}, itr)

            # save the seg_dict as np.array
            np.save(os.path.join(cfg.MODEL_SAVE_DIR, 'seg_dict.npy'), dataloader.dataset.seg_dict)

    # at the end of the training iteration, save the stats, it will automatically overwrite the epoch stats
    saving_state = {
        'net': parameter_source.state_dict(),
        'w': weight.data.cpu(),
        'optimizer': optimizer.state_dict(),
        'IoU_dict': IoU_npl_dict,
        'updated_class': Updated_class_list
    }

    save_path = os.path.join(cfg.MODEL_SAVE_DIR, '%s_%s_%s_checkpoint_itr%d_all.pth' % (
        cfg.MODEL_NAME, cfg.MODEL_BACKBONE, cfg.DATA_NAME, args.TRAIN_ITERATION))

    torch.save(saving_state, save_path)

    if cfg.TRAIN_TBLOG:
        tblogger.close()
    print('%s has been saved' % save_path)
    writelog(cfg, period)

    writer.finalize()


def adjust_lr(optimizer, itr, max_itr, lr_init, power):
    now_lr = lr_init * (1 - itr / (max_itr + 1)) ** power
    optimizer.param_groups[0]['lr'] = now_lr
    optimizer.param_groups[1]['lr'] = 2 * now_lr
    optimizer.param_groups[2]['lr'] = 10 * now_lr
    optimizer.param_groups[3]['lr'] = 20 * now_lr
    return now_lr


def get_params(model, key):
    for m in model.named_modules():
        if key == 'backbone':
            if ('backbone' in m[0]) and isinstance(m[1], (
                    nn.Conv2d, SynchronizedBatchNorm2d, nn.BatchNorm2d, nn.InstanceNorm2d)):
                for p in m[1].parameters():
                    yield p
        elif key == 'cls':
            if ('cls_conv' in m[0]) and isinstance(m[1], (
                    nn.Conv2d, SynchronizedBatchNorm2d, nn.BatchNorm2d, nn.InstanceNorm2d)):
                for p in m[1].parameters():
                    yield p
        elif key == 'others':
            if ('backbone' not in m[0] and 'cls_conv' not in m[0]) and isinstance(m[1], (
                    nn.Conv2d, SynchronizedBatchNorm2d, nn.BatchNorm2d, nn.InstanceNorm2d)):
                for p in m[1].parameters():
                    yield p


if __name__ == '__main__':
    SpawnContext = mp.get_context('spawn')
    train_net()