import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F



def calc_jsd_single(weight, labels1_a, pred, threshold=0.8, Mask_label255_sign='no'):

    Mask_label255 = (labels1_a < 255).float()  # do not compute the area that is irrelavant (dataaug)
    weight_softmax = F.softmax(weight, dim=0)

    criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none')

    loss = criterion(pred * weight_softmax[0], labels1_a)  # * weight_softmax[0]



    prob = F.softmax(pred, dim=1)
    prob = torch.clamp(prob, 1e-7, 1)

    max_probs = torch.amax(prob*Mask_label255.unsqueeze(1), dim=(-3, -2, -1), keepdim=True)  # select according to the pred without the aug area
    mask = max_probs.ge(threshold).float()


    logp = prob.log()

    log_probs = torch.sum(F.kl_div(logp, prob, reduction='none') * mask, dim=1)
    if Mask_label255_sign == 'yes':
        consistency = sum(log_probs)*Mask_label255
    else:
        consistency = sum(log_probs)

    return torch.mean(loss), torch.mean(consistency), consistency, prob


def calc_jsd_multiscale(weight, labels1_a, pred1, pred2, pred3, threshold=0.8, Mask_label255_sign='no'):

    Mask_label255 = (labels1_a < 255).float()  # do not compute the area that is irrelavant (dataaug)  b,h,w
    weight_softmax = F.softmax(weight, dim=0)

    criterion1 = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
    criterion2 = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
    criterion3 = nn.CrossEntropyLoss(ignore_index=255, reduction='none')

    loss1 = criterion1(pred1 * weight_softmax[0], labels1_a)  # * weight_softmax[0]
    loss2 = criterion2(pred2 * weight_softmax[1], labels1_a)  # * weight_softmax[1]
    loss3 = criterion3(pred3 * weight_softmax[2], labels1_a)  # * weight_softmax[2]

    loss = (loss1 + loss2 + loss3)

    probs = [F.softmax(logits, dim=1) for i, logits in enumerate([pred1, pred2, pred3])]

    weighted_probs = [weight_softmax[i] * prob for i, prob in enumerate(probs)]  # weight_softmax[i]*
    mixture_label = (torch.stack(weighted_probs)).sum(axis=0)
    #mixture_label = torch.clamp(mixture_label, 1e-7, 1)  # h,c,h,w
    mixture_label = torch.clamp(mixture_label, 1e-3, 1-1e-3)  # h,c,h,w

    # add this code block for early torch version where torch.amax is not available
    if torch.__version__=="1.5.0" or torch.__version__=="1.6.0":
        _, max_probs = torch.max(mixture_label*Mask_label255.unsqueeze(1), dim=-3, keepdim=True)
        _, max_probs = torch.max(max_probs, dim=-2, keepdim=True)
        _, max_probs = torch.max(max_probs, dim=-1, keepdim=True)
    else:
        max_probs = torch.amax(mixture_label*Mask_label255.unsqueeze(1), dim=(-3, -2, -1), keepdim=True)
    mask = max_probs.ge(threshold).float()


    logp_mixture = mixture_label.log()

    log_probs = [torch.sum(F.kl_div(logp_mixture, prob, reduction='none') * mask, dim=1) for prob in probs]
    if Mask_label255_sign == 'yes':
        consistency = sum(log_probs)*Mask_label255
    else:
        consistency = sum(log_probs)

    return torch.mean(loss), torch.mean(consistency), consistency, mixture_label



def calc_multiscale_backup(weight, seg_backup, pred1, pred2, pred3, mask_seg_prednan, seg_prediction, Lambda_back=0):
    b,_,h,w = pred1.size()
    seg_tempt = torch.zeros((b, 21, h, w), dtype=torch.float)

    seg_tempt[~mask_seg_prednan[:, :, :, :]] = F.softmax(seg_prediction, dim=1)[
        ~mask_seg_prednan[:, :, :, :]]  # b,h,w
    # need to mask out the irrelavant region NaN

    weight_softmax = F.softmax(weight, dim=0)

    criterion1 = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
    criterion2 = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
    criterion3 = nn.CrossEntropyLoss(ignore_index=255, reduction='none')

    # to let the loss depends on the confidence of previous prediction on the background class    seg_tempt[:,0,:,:]
    if Lambda_back == 0:
        loss1 = torch.mean(criterion1(pred1 * weight_softmax[0], seg_backup.to(0)) * seg_tempt[:, 0, :, :].to(0))
        loss2 = torch.mean(criterion2(pred2 * weight_softmax[1], seg_backup.to(0)) * seg_tempt[:, 0, :, :].to(0))
        loss3 = torch.mean(criterion3(pred3 * weight_softmax[2], seg_backup.to(0)) * seg_tempt[:, 0, :, :].to(0))
    else:
        loss1 = torch.mean(criterion1(pred1 * weight_softmax[0], seg_backup.to(0))) * Lambda_back
        loss2 = torch.mean(criterion2(pred2 * weight_softmax[1], seg_backup.to(0))) * Lambda_back
        loss3 = torch.mean(criterion3(pred3 * weight_softmax[2], seg_backup.to(0))) * Lambda_back

    loss = (loss1 + loss2 + loss3)






    return loss