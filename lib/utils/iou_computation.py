import numpy as np



def update_iou_stat(predict, gt, TP, P, T, num_classes = 21):
    """
    :param predict: the pred of each batch,  should be numpy array, after take the argmax   b,h,w
    :param gt: the gt label of the batch, should be numpy array     b,h,w
    :param TP: True positive
    :param P: positive prediction
    :param T: True seg
    :param num_classes: number of classes in the dataset
    :return: TP, P, T
    """
    cal = gt < 255

    mask = (predict == gt) * cal

    for i in range(num_classes):
        P[i] += np.sum((predict == i) * cal)
        T[i] += np.sum((gt == i) * cal)
        TP[i] += np.sum((gt == i) * mask)

    return TP, P, T


def iter_iou_stat(predict, gt, num_classes = 21):
    """
    :param predict: the pred of each batch,  should be numpy array, after take the argmax   b,h,w
    :param gt: the gt label of the batch, should be numpy array     b,h,w
    :param TP: True positive
    :param P: positive prediction
    :param T: True seg
    :param num_classes: number of classes in the dataset
    :return: TP, P, T
    """
    cal = gt < 255

    mask = (predict == gt) * cal

    TP = np.zeros(num_classes)
    P = np.zeros(num_classes)
    T = np.zeros(num_classes)

    for i in range(num_classes):
        P[i] = np.sum((predict == i) * cal)
        T[i] = np.sum((gt == i) * cal)
        TP[i] = np.sum((gt == i) * mask)

    return np.array([TP, P, T])


def compute_iou(TP, P, T, num_classes = 21):
    """
    :param TP:
    :param P:
    :param T:
    :param num_classes: number of classes in the dataset
    :return: IoU
    """
    IoU = []
    for i in range(num_classes):
        IoU.append(TP[i] / (T[i] + P[i] - TP[i] + 1e-10))
    return IoU


def update_fraction_batchwise(mask, gt, fraction, num_classes = 21):
    """
    :param mask: True when belong to subgroup (memorized, correct, others) which we want to calculate fraction on
    :param gt: the gt label of the batch, numpy array
    :param fraction: fraction of pixels in the subgroup
    :param num_classes: number of classes in the dataset
    :return: updated fraction
    """
    cal = gt < 255

    for i in range(num_classes):
        fraction[i] += np.sum((mask * (gt == i) * cal))/np.sum((gt == i) * cal)

    return fraction


def update_fraction_instancewise(mask, gt, fraction, num_classes = 21):
    """
    :param mask: True when belong to subgroup (memorized, correct, others) which we want to calculate fraction on
    :param gt: the gt label of the batch, numpy array
    :param fraction: fraction of pixels in the subgroup
    :param num_classes: number of classes in the dataset
    :return: updated fraction
    """
    # np.sum((gt == i) * cal maybe a nan value, can't do that
    cal = gt < 255

    for i in range(num_classes):
        fraction[i] += np.mean(np.sum((mask * (gt == i) * cal), axis= (-2,-1))/np.sum((gt == i) * cal, axis= (-2,-1)))

    return fraction

def update_fraction_pixelwise(mask, gt, abs_num_and_total, num_classes = 21):
    """
    :param mask: True when belong to subgroup (memorized, correct, others) which we want to calculate fraction on
    :param gt: the gt label of the batch, numpy array
    :param abs_num_and_total: the absolute number of pixel belong to the mask and the total num of pixels [abs_num, pixel_num]
    :param num_classes: number of classes in the dataset
    :return: updated fraction
    """
    cal = gt < 255

    for i in range(num_classes):
        abs_num_and_total[i][0] += np.sum(mask * (gt == i) * cal)
        abs_num_and_total[i][1] += np.sum((gt == i) * cal)


    return abs_num_and_total

def iter_fraction_pixelwise(mask, gt, num_classes = 21):
    """
    :param mask: True when belong to subgroup (memorized, correct, others) which we want to calculate fraction on
    :param gt: the gt label of the batch, numpy array
    :param num_classes: number of classes in the dataset
    :return: updated fraction
    """
    cal = gt < 255

    abs_num_and_total = np.zeros((num_classes,2))

    for i in range(num_classes):
        abs_num_and_total[i][0] += np.sum(mask * (gt == i) * cal)
        abs_num_and_total[i][1] += np.sum((gt == i) * cal)


    return abs_num_and_total



def get_mask(gt_np, label_np, pred_np):
    """

    Args:
        gt_np: the GT label
        label_np: the CAM pseudo label
        pred_np: the prediction

    Returns: the mask of different type

    """
    wrong_mask_correct = (gt_np != label_np) & (pred_np == gt_np)
    wrong_mask_memorized = (gt_np != label_np) & (pred_np == label_np)
    wrong_mask_others = (gt_np != label_np) & (pred_np != gt_np) & (pred_np != label_np)
    clean_mask_correct = (gt_np == label_np) & (pred_np == gt_np)
    clean_mask_incorrect = (gt_np == label_np) & (pred_np != gt_np)

    return (wrong_mask_correct,wrong_mask_memorized,wrong_mask_others,clean_mask_correct,clean_mask_incorrect)