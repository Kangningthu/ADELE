from scipy.optimize import curve_fit
import numpy as np

def curve_func(x, a, b, c):
    return a *(1-np.exp( -1/c * x**b  ))


def fit(func, x, y):
    popt, pcov = curve_fit(func, x, y, p0 =(1,1,1), method= 'trf', sigma = np.geomspace(1,.1,len(y)), absolute_sigma=True, bounds= ([0,0,0],[1,1,np.inf]) )
    return tuple(popt)


def derivation(x, a, b, c):
    x = x + 1e-6 # numerical robustness
    return a * b * 1/c * np.exp(-1/c * x**b) * (x**(b-1))


def label_update_epoch(ydata_fit, n_epoch = 16, threshold = 0.9, eval_interval = 100, num_iter_per_epoch= 10581/10):
    xdata_fit = np.linspace(0, len(ydata_fit)*eval_interval/num_iter_per_epoch, len(ydata_fit))
    a, b, c = fit(curve_func, xdata_fit, ydata_fit)
    epoch = np.arange(1, n_epoch)
    y_hat = curve_func(epoch, a, b, c)
    relative_change = abs(abs(derivation(epoch, a, b, c)) - abs(derivation(1, a, b, c)))/ abs(derivation(1, a, b, c))
    relative_change[relative_change > 1] = 0
    update_epoch = np.sum(relative_change <= threshold) + 1
    return update_epoch#, a, b, c

def if_update(iou_value, current_epoch, n_epoch = 16, threshold = 0.90, eval_interval=1, num_iter_per_epoch=1):
    # check iou_value
    start_iter = 0
    print("len(iou_value)=",len(iou_value))
    for k in range(len(iou_value)-1):
        if iou_value[k+1]-iou_value[k] < 0.1:
            start_iter = max(start_iter, k + 1)
        else:
            break
    shifted_epoch = start_iter*eval_interval/num_iter_per_epoch
    #cut out the first few entries
    iou_value = iou_value[start_iter: ]
    update_epoch = label_update_epoch(iou_value, n_epoch = n_epoch, threshold=threshold, eval_interval=eval_interval, num_iter_per_epoch=num_iter_per_epoch)
    # Shift back
    update_epoch = shifted_epoch + update_epoch
    return current_epoch >= update_epoch#, update_epoch


def merge_labels_with_skip(original_labels, model_predictions, need_label_correction_dict, conf_threshold=0.8, logic_255=False,class_constraint=True, conf_threshold_bg = 0.95):


    new_label_dict = {}
    update_list = []
    for c in need_label_correction_dict:
        if need_label_correction_dict[c]:
            update_list.append(c)


    for pid in model_predictions:
        pred_prob = model_predictions[pid]
        pred = np.argmax(pred_prob, axis=0)
        label = original_labels[pid]

        # print(np.unique(label))
        # print(update_list)
        # does not belong to the class that need to be updated, then we do not need the following updating process
        if set(np.unique(label)).isdisjoint(set(update_list)):
            new_label_dict[pid] = label
            continue


        # if the prediction is confident
        # confident = np.max(pred_prob, axis=0) > conf_threshold

        # if the prediction is confident
        # code support different threshold for foreground and background,
        # during the experiment, we always set them to be the same for simplicity
        confident = (np.max(pred_prob[1:], axis=0) > conf_threshold) |(pred_prob[0] > conf_threshold_bg)

        # before update: only class that need correction will be replaced
        belong_to_correction_class = label==0
        for c in need_label_correction_dict:
            if need_label_correction_dict[c]:
                belong_to_correction_class |= (label==c)

        # after update: only pixels that will be flipped to the allowed classes will be updated
        after_belong = pred==0
        for c in need_label_correction_dict:
            if need_label_correction_dict[c]:
                after_belong |= (pred==c)

        # combine all three masks together
        replace_flag = confident & belong_to_correction_class & after_belong


        # the class constraint
        if class_constraint:
            unique_class = np.unique(label)
            # print(unique_class)
            # indx = torch.zeros((h, w), dtype=torch.long)
            class_constraint_indx = (pred==0)
            for element in unique_class:
                class_constraint_indx = class_constraint_indx | (pred == element)


            replace_flag = replace_flag & (class_constraint_indx != 0)


        # replace with the new label
        next_label = np.where(replace_flag, pred, label).astype("int32")

        # logic 255:
        # - rule# 1: if label[i,j] != 0, and pred[i,j] = 0, then next_label[i,j] = 255
        # - rule# 2: if label[i,j] = 255 and pred[i,j] != 0 and confident, then next_label[i,j] = pred[i,j]
        # rule 2 is already enforced above, don't need additional code
        if logic_255:
            rule_1_flag = (label != 0) & (pred == 0)
            next_label = np.where(rule_1_flag, np.ones(next_label.shape)*255, next_label).astype("int32")

        new_label_dict[pid] = next_label

    return new_label_dict


