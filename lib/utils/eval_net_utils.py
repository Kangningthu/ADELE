import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import cv2
from .imutils import img_denorm
from .DenseCRF import dense_crf
import pickle
import os
import torch.multiprocessing as mp

def eval_net_multiprocess(SpawnContext, net1, net2, IoU_npl_indx, train_dataloader, eval_dataloader1,
                          eval_dataloader2, momentum=0.3, scale_index=0, flip='no',
                          scalefactor=1.0, CRF_post='no', tempt_save_root='.',update_all_bg_img=True,t_eval=3):
    net1.eval()
    net2.eval()
    if torch.cuda.device_count() > 1:


        seg_dict_copy = train_dataloader.dataset.seg_dict.copy()
        p1 = SpawnContext.Process(target = eval_net_bs_one, args=(torch.device(0), net1, IoU_npl_indx, eval_dataloader1,seg_dict_copy, momentum, scale_index, flip, scalefactor, CRF_post, tempt_save_root, 'eval_dict_tempt1.npy',update_all_bg_img,t_eval))
        p2 = SpawnContext.Process(target = eval_net_bs_one, args=(torch.device(1), net2, IoU_npl_indx, eval_dataloader2,seg_dict_copy, momentum, scale_index, flip, scalefactor, CRF_post, tempt_save_root, 'eval_dict_tempt2.npy',update_all_bg_img,t_eval))

        p1.start()
        p2.start()

        p1.join()
        p2.join()


        tempt = np.load(os.path.join(tempt_save_root, 'eval_dict_tempt1.npy'), allow_pickle=True)
        prev_pred_dict = tempt[()]

        tempt2 = np.load(os.path.join(tempt_save_root, 'eval_dict_tempt2.npy'), allow_pickle=True)
        prev_pred_dict2 = tempt2[()]

        prev_pred_dict.update(prev_pred_dict2)
        train_dataloader.dataset.prev_pred_dict = prev_pred_dict

        os.remove(os.path.join(tempt_save_root, 'eval_dict_tempt1.npy'))
        os.remove(os.path.join(tempt_save_root, 'eval_dict_tempt2.npy'))
        del seg_dict_copy



    return


def eval_net_bs_one(device, net, IoU_npl_indx, eval_dataloader, seg_dict_copy, momentum=0.3, scale_index=0, flip='no', scalefactor=1.0, CRF_post='no',tempt_save_root='.', save_name='eval_dict_tempt1.npy', update_all_bg_img=False, t_eval=1.0):
    # net.eval()
    #scale_index = 2 # currently only support this version, improve later
    if scale_index==0:
        TEST_MULTISCALE = [0.75, 1.0, 1.5]
    elif scale_index==1:
        TEST_MULTISCALE = [0.5, 1.0, 1.75]
    elif scale_index==2:
        TEST_MULTISCALE = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    elif scale_index==3:
        TEST_MULTISCALE = [0.7, 1.0, 1.5]
    elif scale_index==4:
        TEST_MULTISCALE = [0.5, 0.75, 1.0, 1.25, 1.5]
    elif scale_index==5:
        TEST_MULTISCALE = [1.0]
    # print('eval_with_onebyone')
    prev_pred_dict = {}
    with tqdm(total=len(eval_dataloader)) as pbar:
        with torch.no_grad():
            for i_batch, sample in enumerate(eval_dataloader):
                # print(sample['batch_idx'])
                # seg_labels = sample['segmentation']

                seg_labels = seg_dict_copy[eval_dataloader.dataset.ori_indx_list[sample['batch_idx']]]
                # if they are not disjoint, we should evaluate it
                if set(np.unique(seg_labels[0].cpu().numpy())).isdisjoint(set(IoU_npl_indx[1:])):

                    if update_all_bg_img and not (set(np.unique(seg_labels[0].numpy())) - set(np.array([0, 255]))):
                        # only the background in the pseudo label, then this picture will still be evaluated
                        pass
                    else:
                        # skip this one
                        continue

                inputs = sample['image']
                n, c, h, w = inputs.size()   # 1,c,h,w
                result_list =[]
                image_multiscale = []
                for rate in TEST_MULTISCALE:
                    inputs_batched = sample['image_%f' % rate]
                    image_multiscale.append(inputs_batched)
                    if flip!='no':
                        image_multiscale.append(torch.flip(inputs_batched, [3]))
                for img in image_multiscale:
                    result = net(img.to(device))
                    result_list.append(result.cpu())
                    img.cpu()

                for i in range(len(result_list)):
                    result_seg = F.interpolate(result_list[i], (h,w), mode='bilinear', align_corners=True)
                    if i % 2 == 1 and flip!='no':
                        result_seg = torch.flip(result_seg, [3])
                    result_list[i] = result_seg
                prob_seg = torch.stack(result_list, dim=0)   # 12, 1, c,h,w
                prob_seg = F.softmax(torch.mean(prob_seg/t_eval, dim=0, keepdim=False), dim=1) # 1,c,h,w
                #prob_seg = torch.clamp(prob_seg, 1e-7, 1)
                # do the CRF
                if CRF_post !='no':
                    prob = prob_seg.cpu().numpy()  # 1,c,h,w
                    img_batched = img_denorm(sample['image'][0].numpy()).astype(np.uint8)
                    prob = dense_crf(prob[0], img_batched, n_classes=21, n_iters=1)
                    prob_seg = torch.from_numpy(prob.astype(np.float32))
                    result = prob_seg.unsqueeze(dim=0)  # 1,c,h,w
                else:
                    result = prob_seg.cpu()  # 1,c,h,w

                result_argmax = torch.argmax(result,dim=1)  # 1,c,h,w  the pred argmax label
                result_max_prob, _ = torch.max(result, dim=1) # 1,c,h,w  the max probability
                for batch_idx in sample['batch_idx'].numpy():
                    # prev_pred_dict[batch_idx] = result
                    prev_pred_dict[eval_dataloader.dataset.ori_indx_list[batch_idx]]= (result_argmax, result_max_prob)
                pbar.set_description("Correcting Labels ")
                pbar.update(1)

    np.save(os.path.join(tempt_save_root, save_name), prev_pred_dict)

