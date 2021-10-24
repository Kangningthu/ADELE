import pandas as pd
import numpy as np
import torch, os, argparse, logging, random, sys, pickle, time
from brat import loading, unet_model, brat_util, label_correction
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from lib.utils import JSD_loss, iou_computation

def dice_coefficient(pred, label, epsilon=1e-6):
    """
    Function that calculates the dice coefficient
    :param pred: N,C,H,W
    :param label: N,C,H,W
    :param epsilon:
    :return: N,C
    """
    # populate the output holder
    numerator = (pred * label).sum(dim=-1).sum(dim=-1) * 2
    denominator = pred.pow(2).sum(dim=-1).sum(dim=-1) + label.pow(2).sum(dim=-1).sum(dim=-1) + epsilon
    dice_val = numerator/denominator
    # return NaN for example where a class is not present
    no_show_indicator = label.sum(dim=-1).sum(dim=-1)==0
    return torch.where(no_show_indicator, torch.ones(dice_val.size()).to(dice_val.device) * float('nan'), dice_val)


def avg_masked_dice(pred_prob, dice_label):
    # loss = 1 - dice
    dice = dice_coefficient(pred_prob, dice_label)
    one_minus_dice = 1.0 - dice
    # impute NaN to 0 and then calculate average that ignore NaN
    dice_imputed = torch.where(torch.isnan(one_minus_dice), torch.zeros(one_minus_dice.size()).to(one_minus_dice.device), one_minus_dice)
    dice_imputed_avg = dice_imputed.sum(dim=1) / (1.0 - torch.isnan(dice).float()).sum(dim=1)
    return dice_imputed_avg.mean()


def dice_loss(pred, label, ignore_background=False):
    # prepare label and probability
    with torch.no_grad():
        dice_label = torch.cat([(label == 0).float().unsqueeze(1),
                                (label == 1).float().unsqueeze(1),
                                (label == 2).float().unsqueeze(1),
                                (label == 3).float().unsqueeze(1),
                                (label == 4).float().unsqueeze(1)],
                               dim=1)
        if ignore_background:
            dice_label = torch.cat([(label == 1).float().unsqueeze(1),
                                    (label == 2).float().unsqueeze(1),
                                    (label == 3).float().unsqueeze(1),
                                    (label == 4).float().unsqueeze(1)],
                                   dim=1)
    if ignore_background:
        pred_prob = F.softmax(pred[:,1:,:,:], dim=1)
    else:
        pred_prob = F.softmax(pred, dim=1)
    # dice loss = avg_masked_dice(prob, dice) + avg_masked_dice(1.0-prob, 1.0-dice)
    return avg_masked_dice(pred_prob, dice_label) + avg_masked_dice(1.0-pred_prob, 1.0-dice_label)


def eval_model_at_scale(img, scale_factor, model, flip=False):
    """
    Evaluate the model on a different scale, make a prediction, and then transform the scale back.
    """
    # interpolate image
    if scale_factor != 1.0:
        input = F.interpolate(img, scale_factor=scale_factor, mode="bilinear", align_corners=True,
                                    recompute_scale_factor=True)
    else:
        input = img
    # flip the image if necessary
    if flip:
        input = torch.flip(input, dims=[-1])
    # make prediction
    pred = model(input)
    # interpolate back to the original resolution
    _, _, h, w = img.size()
    # flip back
    if flip:
        pred = torch.flip(pred, dims=[-1])
    # interpolate back
    if scale_factor != 1.0:
        pred =  F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)
    return pred


def eval_model_multiple_scales(img, model, scales, flip):
    # make a prediction at every scale
    all_pred_list = []
    for scale in scales:
        all_pred_list.append(eval_model_at_scale(img, scale, model, False).unsqueeze(-1))
        if flip:
            all_pred_list.append(eval_model_at_scale(img, scale, model, True).unsqueeze(-1))
    # turn into probability
    prob = F.softmax(torch.cat(all_pred_list, dim=-1), dim=1)
    # average across scales
    avg_across_scales_prob = prob.mean(-1)
    return avg_across_scales_prob
def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)

def run_experiment(parameters):
    # create model dir
    if parameters['save_dir'] is not None:
        model_dir = os.path.join(parameters['save_dir'], parameters['model_name'])
        assert not os.path.exists(model_dir), "This model directory already exists"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    else:
        model_dir = None

    # set random seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    random.seed(parameters["seed"])
    torch.manual_seed(parameters["seed"])
    torch.cuda.manual_seed(parameters["seed"])
    np.random.seed(parameters["seed"])

    # create logger
    if model_dir is not None:
        logging.basicConfig(filename=os.path.join(model_dir, "log"), level=logging.INFO)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    else:
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.info("model_dir = {0}".format(model_dir))
    logging.info("parameters = {0}".format(parameters))
    logger = logging.getLogger(__name__)
    global_doc = brat_util.DocumentUnit(["epoch", "phase", "time",
                                         "dice_1_avg", "dice_2_avg", "dice_3_avg", "dice_4_avg",
                                         "dice_1_std", "dice_2_std", "dice_3_std", "dice_4_std",
                                         "iou_0", "iou_1", "iou_2", "iou_3", "iou_4",
                                         "iou_noise_0", "iou_noise_1", "iou_noise_2", "iou_noise_3", "iou_noise_4",
                                         "iou_clean_0", "iou_clean_1", "iou_clean_2", "iou_clean_3", "iou_clean_4",
                                         "iou_update_0", "iou_update_1", "iou_update_2", "iou_update_3", "iou_update_4",
                                         "iou_agree_0", "iou_agree_1", "iou_agree_2", "iou_agree_3", "iou_agree_4"])

    # create model
    model = unet_model.UNet(n_channels=1, n_classes=5)
    device = torch.device("cuda")
    model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model = model.to(device)

    # data loader
    with open(parameters["data_list"], "rb") as f:
        # tr_dl, val_dl, ts_dl = pickle.load(f)
        tr_dl, ts_dl,val_dl = pickle.load(f)
    tr_cache_dir = None if parameters["cache_dir"] is None else os.path.join(parameters["cache_dir"], "tr.pkl")
    val_cache_dir = None if parameters["cache_dir"] is None else os.path.join(parameters["cache_dir"], "val.pkl")
    ts_cache_dir = None if parameters["cache_dir"] is None else os.path.join(parameters["cache_dir"], "ts.pkl")

    need_aug = not parameters["no_augmentation"]
    tr_ds = loading.SegTHORDataset(parameters, tr_dl, augmentation=need_aug, noise_label=parameters["noise_label"],
                                   cache_dir=tr_cache_dir, noise_level=parameters["noise_level"])
    val_ds = loading.SegTHORDataset(parameters, val_dl, augmentation=False,cache_dir=val_cache_dir)
    ts_ds = loading.SegTHORDataset(parameters, ts_dl, augmentation=False, cache_dir=ts_cache_dir)
    eval_ds = loading.SegTHORDataset(parameters, tr_dl, augmentation=False, cache_dir=tr_cache_dir)



    tr_loader = DataLoader(dataset=tr_ds, batch_size=parameters["batch_size"],
                           shuffle=True, num_workers=17, pin_memory=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(dataset=val_ds, batch_size=parameters["batch_size"],
                           shuffle=False, num_workers=17, pin_memory=True, worker_init_fn=worker_init_fn)
    ts_loader = DataLoader(dataset=ts_ds, batch_size=parameters["batch_size"],
                           shuffle=False, num_workers=17, pin_memory=True, worker_init_fn=worker_init_fn)
    eval_loader = DataLoader(dataset=eval_ds, batch_size=parameters["batch_size"],
                           shuffle=False, num_workers=17, pin_memory=True, worker_init_fn=worker_init_fn)

    # optimizer and loss functions
    optimizer = torch.optim.SGD(model.parameters(), lr=parameters["lr"] , momentum=0.99)
    # multi-label dice loss
    if parameters["loss"] == "ce":
        if parameters["background_weight"] is not None:
            criterion = nn.CrossEntropyLoss(weight=torch.Tensor([parameters["background_weight"], 1, 1, 1, 1]).to(device))
        elif parameters["balance"]:
            criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1.01729696, 1441.98067, 79.3493525, 1792.17319, 317.571485]).to(device))
        else:
            criterion = nn.CrossEntropyLoss(ignore_index=255)
    elif parameters["loss"] == "dice":
        criterion = lambda x,y: dice_loss(x,y,parameters["ignore_background"])

    # start experiments
    need_label_correction_dict = {1:False, 2:False, 3:False, 4:False}
    already_label_correction_dict = {1:False, 2:False, 3:False, 4:False}
    for epoch_number in range(1, parameters["number_epochs"] + 1):
        if parameters["phase"]["training"]:
            tr_pred = epoch(model, tr_loader, criterion, optimizer,
                  parameters, device, "training", epoch_number, model_dir, global_doc, save_res=True)

            # if label correction is required for next epoch
            if parameters["label_correction"]:
                # decide if the labels need to be corrected for the next epoch
                # once the label correction is on for this class, it will always be on
                performance_df = global_doc.form_df()
                tr_df = performance_df[performance_df["phase"] == "training"]
                for c in [1, 2, 3, 4]:
                    if not need_label_correction_dict[c]:
                        need_label_correction_dict[c] = label_correction.if_update(tr_df["iou_{}".format(c)].values, epoch_number - 1, n_epoch=parameters["number_epochs"],
                                                                     threshold=parameters["r"])
                        #need_label_correction_dict[c] = label_correction.if_update_test(c, epoch_number)

                # reset already_label_correction_dict
                # correct_freq is None = correct every epoch
                # correct_freq < 0: only correct once
                # correct_freq = 5: correct every 5 epoch
                if parameters["correct_freq"] is not None and parameters["correct_freq"] > 0 and epoch_number % \
                        parameters["correct_freq"] == 0:
                    already_label_correction_dict = {1: False, 2: False, 3: False, 4: False}

                # if correct_once, prevent the model from correction if already corrected
                if parameters["correct_freq"] is not None:
                    for c in [1,2,3,4]:
                        if already_label_correction_dict[c]:
                            need_label_correction_dict[c] = False

                # if any class needs to be corrected, recompute labels for all training examples
                if True in need_label_correction_dict.values():
                    logging.info("start label correction")
                    # recompute model outputs using multiple scales
                    # merge labels
                    new_labels = label_correction.merge_labels_with_skip(tr_loader.dataset.cache_label, tr_pred,
                                            need_label_correction_dict, conf_threshold=parameters["conf_threshold"],
                                            logic_255=parameters["logic_255"],conf_threshold_bg=parameters["conf_threshold_bg"])
                    temp_dir = parameters["save_dir"]
                    save_dir = os.path.join(temp_dir, parameters["model_name"])
                    if not os.path.exists(save_dir):
                        os.mkdir(save_dir)
                    with open(os.path.join(save_dir, "new_labels_epoch_{}.pkl".format(epoch_number)), "wb") as f:
                        pickle.dump(new_labels, f)
                    with open(os.path.join(save_dir, "tr_pred_epoch_{}.pkl".format(epoch_number)), "wb") as f:
                        pickle.dump(tr_pred, f)
                    # reset labels for next iterations
                    tr_loader.dataset.reset_labels(new_labels)

                # update already_label_correction_dict
                for c in [1, 2, 3, 4]:
                    if not already_label_correction_dict[c]:
                        already_label_correction_dict[c] = need_label_correction_dict[c]

        if parameters["phase"]["validation"]:
            epoch(model, val_loader, criterion, optimizer,
                  parameters, device, "validation", epoch_number, model_dir, global_doc, save_res=True)
        if parameters["phase"]["test"]:
            epoch(model, ts_loader, criterion, optimizer,
                  parameters, device, "test", epoch_number, model_dir, global_doc, save_res=True)


def epoch(model, data_loader, criterion, optimizer,
          parameters, device, phase, epoch_number,
          model_dir, global_document_unit, save_res=False, multiscale=False):

    start_epoch_time = time.time()
    logging.info("Start {0} epoch {1}".format(phase, epoch_number))
    # create folder for this epoch
    if model_dir is not None:
        epoch_dir = os.path.join(model_dir, "epoch_{0}".format(epoch_number))
        if not os.path.exists(epoch_dir):
            os.mkdir(epoch_dir)

    # create epoch document unit
    document_columns = ["img", "dice_1", "dice_2", "dice_3", "dice_4"]
    epoch_document_unit = brat_util.DocumentUnit(document_columns)
    if save_res:
        save_pred = {}

    # run-time statistics
    runtime_profiler = brat_util.RuntimeProfiler()
    total_imgs = 0

    # update model phase
    if phase == "training":
        model.train()
    elif phase in ["validation", "test", "eval"]:
        model.eval()

    # start the epoch
    iou_whole_img = [np.zeros(5), np.zeros(5), np.zeros(5)]
    iou_clean = [np.zeros(5), np.zeros(5), np.zeros(5)]
    iou_noise = [np.zeros(5), np.zeros(5), np.zeros(5)]
    iou_agree_clean = [np.zeros(5), np.zeros(5), np.zeros(5)]
    iou_updated_label = [np.zeros(5), np.zeros(5), np.zeros(5)]

    for i, (imgs, labels, original_noisy_labels, clean_labels, filenames) in enumerate(data_loader):


        # load data
        input_img_variable = Variable(imgs.to(device)).float()
        input_label_variable = Variable(labels.to(device)).long().squeeze(1)
        runtime_profiler.tik("data_loading")
        iou_computation.update_iou_stat(input_label_variable.data.cpu().numpy(), clean_labels.cpu().numpy()[:, 0, :, :],
                                        iou_updated_label[0], iou_updated_label[1], iou_updated_label[2], 5)
        # forward propagation
        if phase == "training":
            pred_is_prob = False
            pred = model(input_img_variable)
        elif phase in ["validation", "test", "eval"]:
            with torch.no_grad():
                if multiscale:
                    pred_is_prob = True
                    pred = eval_model_multiple_scales(input_img_variable, model, scales=[0.7, 1.0, 1.5], flip=True)
                else:
                    pred_is_prob = False
                    pred = model(input_img_variable)
        runtime_profiler.tik("forward")

        # calculate loss
        if phase == "training":
            if parameters["jsd_lambda"] != 0:
                # create image+ and image-
                pred_small_rescale = eval_model_at_scale(input_img_variable, 0.7, model)
                pred_large_rescale = eval_model_at_scale(input_img_variable, 1.5, model)
                # weight is set to all 1
                weight = nn.Parameter(torch.Tensor(3)).to(device)
                weight.data.fill_(1)
                # calculate jsd loss
                loss_ce, consistency, variance, mixture_label = JSD_loss.calc_jsd_multiscale(weight, input_label_variable,
                                                                        pred_small_rescale, pred, pred_large_rescale,threshold=parameters["rho"])
                loss = (parameters["jsd_lambda"] * consistency + loss_ce) / 3
                # logging.info("jsd_loss = {0}".format(consistency))
            else:
                loss = criterion(pred, input_label_variable)
            # logging.info("loss = {0}".format(loss))
            # backward propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        runtime_profiler.tik("backward")

        # log model behaviours
        epoch_document_unit.add_values("img", filenames)
        if pred_is_prob:
            pred_prob = pred
        else:
            pred_prob = F.softmax(pred, dim=1)
        pred_np = pred_prob.data.cpu().numpy()
        if save_res:
            for i in range(len(filenames)):
                save_pred[filenames[i]] = pred_np[i, :, :, :]

        # dice
        dice_label = torch.cat([(input_label_variable==0).float().unsqueeze(1),
                      (input_label_variable == 1).float().unsqueeze(1),
                      (input_label_variable == 2).float().unsqueeze(1),
                      (input_label_variable == 3).float().unsqueeze(1),
                      (input_label_variable == 4).float().unsqueeze(1)],
                               dim=1)
        dice = dice_coefficient(pred_prob, dice_label).data.cpu().numpy()
        epoch_document_unit.add_values("dice_1", list(dice[:,1]))
        epoch_document_unit.add_values("dice_2", list(dice[:,2]))
        epoch_document_unit.add_values("dice_3", list(dice[:,3]))
        epoch_document_unit.add_values("dice_4", list(dice[:,4]))

        # update IOU
        pred_np_discrete = torch.argmax(pred,dim=1).data.cpu().numpy()
        iou_computation.update_iou_stat(pred_np_discrete,
                                        input_label_variable.data.cpu().numpy(),
                                        iou_whole_img[0], iou_whole_img[1], iou_whole_img[2], 5)
        # clean IOU: IOU(pred_prob, clean_labels[diff_pixel_set])
        # noisy IOU: IOU(pred_prob, noisy_labels[diff_pixel_set])
        # diff_pixel_set = a mask where noisy labels != clean labels
        diff_pixel_set = original_noisy_labels != clean_labels
        is_diff_flag = (diff_pixel_set.sum(dim=-1).sum(dim=-1).sum(dim=-1) > 0).data.cpu().numpy()
        diff_pixel_set = diff_pixel_set.data.cpu().numpy()
        for j in range(len(is_diff_flag)):
            if is_diff_flag[j]:
                mask = diff_pixel_set[j,0,:,:]
                # calculate clean IOU: green curve
                masked_clean_label = np.copy(clean_labels[j,0,:,:])
                masked_clean_label[~mask] = 255
                iou_computation.update_iou_stat(pred_np_discrete[j,:,:], masked_clean_label,
                                                iou_clean[0], iou_clean[1], iou_clean[2], 5)

                # calculate clean IOU: red curve
                masked_noisy_label = np.copy(original_noisy_labels[j, 0, :, :])
                masked_noisy_label[~mask] = 255
                iou_computation.update_iou_stat(pred_np_discrete[j, :, :], masked_noisy_label,
                                                iou_noise[0], iou_noise[1], iou_noise[2], 5)
        # iou_agree_clean: IOU(pred_prob, clean_labels[agree_pixel_set])
        agree_pixel_set = (original_noisy_labels == clean_labels).data.cpu().numpy()
        for j in range(agree_pixel_set.shape[0]):
            mask = agree_pixel_set[j,0,:,:]
            masked_clean_label = np.copy(clean_labels[j,0,:,:])
            masked_clean_label[~mask] = 255
            iou_computation.update_iou_stat(pred_np_discrete[j, :, :], masked_clean_label,
                                            iou_agree_clean[0], iou_agree_clean[1], iou_agree_clean[2], 5)

        # compute time
        total_imgs += input_img_variable.size()[0]
        logging.info("minibatch_number = {0}, {1}/{2} done".format(i, total_imgs, len(data_loader.dataset)))
        runtime_profiler.tik("report")

    # report performance
    epoch_iou = iou_computation.compute_iou(iou_whole_img[0], iou_whole_img[1], iou_whole_img[2], 5)
    epoch_iou_clean = iou_computation.compute_iou(iou_clean[0], iou_clean[1], iou_clean[2], 5)
    epoch_iou_noise = iou_computation.compute_iou(iou_noise[0], iou_noise[1], iou_noise[2], 5)
    epoch_iou_update = iou_computation.compute_iou(iou_updated_label[0], iou_updated_label[1], iou_updated_label[2], 5)
    epoch_iou_agree = iou_computation.compute_iou(iou_agree_clean[0], iou_agree_clean[1], iou_agree_clean[2], 5)

    global_document_unit.add_values("epoch", [epoch_number])
    global_document_unit.add_values("phase", [phase])
    global_document_unit.add_values("dice_1_avg", [np.nanmean(epoch_document_unit.data_dict["dice_1"])])
    global_document_unit.add_values("dice_1_std", [np.nanstd(epoch_document_unit.data_dict["dice_1"])])
    global_document_unit.add_values("dice_2_avg", [np.nanmean(epoch_document_unit.data_dict["dice_2"])])
    global_document_unit.add_values("dice_2_std", [np.nanstd(epoch_document_unit.data_dict["dice_2"])])
    global_document_unit.add_values("dice_3_avg", [np.nanmean(epoch_document_unit.data_dict["dice_3"])])
    global_document_unit.add_values("dice_3_std", [np.nanstd(epoch_document_unit.data_dict["dice_3"])])
    global_document_unit.add_values("dice_4_avg", [np.nanmean(epoch_document_unit.data_dict["dice_4"])])
    global_document_unit.add_values("dice_4_std", [np.nanstd(epoch_document_unit.data_dict["dice_4"])])
    global_document_unit.add_values("iou_0", [epoch_iou[0]])
    global_document_unit.add_values("iou_1", [epoch_iou[1]])
    global_document_unit.add_values("iou_2", [epoch_iou[2]])
    global_document_unit.add_values("iou_3", [epoch_iou[3]])
    global_document_unit.add_values("iou_4", [epoch_iou[4]])
    global_document_unit.add_values("iou_clean_0", [epoch_iou_clean[0]])
    global_document_unit.add_values("iou_clean_1", [epoch_iou_clean[1]])
    global_document_unit.add_values("iou_clean_2", [epoch_iou_clean[2]])
    global_document_unit.add_values("iou_clean_3", [epoch_iou_clean[3]])
    global_document_unit.add_values("iou_clean_4", [epoch_iou_clean[4]])
    global_document_unit.add_values("iou_noise_0", [epoch_iou_noise[0]])
    global_document_unit.add_values("iou_noise_1", [epoch_iou_noise[1]])
    global_document_unit.add_values("iou_noise_2", [epoch_iou_noise[2]])
    global_document_unit.add_values("iou_noise_3", [epoch_iou_noise[3]])
    global_document_unit.add_values("iou_noise_4", [epoch_iou_noise[4]])
    global_document_unit.add_values("iou_update_0", [epoch_iou_update[0]])
    global_document_unit.add_values("iou_update_1", [epoch_iou_update[1]])
    global_document_unit.add_values("iou_update_2", [epoch_iou_update[2]])
    global_document_unit.add_values("iou_update_3", [epoch_iou_update[3]])
    global_document_unit.add_values("iou_update_4", [epoch_iou_update[4]])
    global_document_unit.add_values("iou_agree_0", [epoch_iou_agree[0]])
    global_document_unit.add_values("iou_agree_1", [epoch_iou_agree[1]])
    global_document_unit.add_values("iou_agree_2", [epoch_iou_agree[2]])
    global_document_unit.add_values("iou_agree_3", [epoch_iou_agree[3]])
    global_document_unit.add_values("iou_agree_4", [epoch_iou_agree[4]])
    global_document_unit.add_values("time", [time.time()-start_epoch_time])

    # save epoch document unit and update the global document unit
    if model_dir is not None:
        epoch_dir = os.path.join(model_dir, "epoch_{}".format(epoch_number))
        if not os.path.exists(epoch_dir):
            os.mkdir(epoch_dir)
        # if save_res:
        #     with open(os.path.join(epoch_dir, "{}_pred.pkl".format(phase)), "wb") as f:
        #         pickle.dump(save_pred, f)
        epoch_document_unit.to_csv(os.path.join(epoch_dir, "{0}.csv".format(phase)))
        global_document_unit.to_csv(os.path.join(model_dir, "performance.csv".format(phase)))
        # save the model at every epoch
        if phase == "training":
            model_file_name = os.path.join(epoch_dir, 'model.ckpt')
            optimizer_file_name = os.path.join(epoch_dir, 'optimizer.ckpt')
            torch.save(model.module.state_dict(), model_file_name)
            torch.save(optimizer.state_dict(), optimizer_file_name)
            logging.info("model saved to {0}".format(model_file_name))

    # epoch logging
    logging.info("{0} epoch {1} finished takes {2} seconds".format(phase, epoch_number, time.time()-start_epoch_time))
    logging.info(runtime_profiler.report_avg())

    if save_res:
        return save_pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SegTHOR 2020 Challange")
    parser.add_argument("--data-list", type=str, default="")
    parser.add_argument("--data-dir", type=str, default="")
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--noise-label", type=str, default='final')
    parser.add_argument("--noise-level", type=float, default=1)
    parser.add_argument("--label-correction", action="store_true", default=False)
    parser.add_argument("--correct-freq", type=int, default=10)
    parser.add_argument("--logic255", action="store_true", default=False)

    # label correction related
    parser.add_argument("--tau_fg", type=float, default=0.7, help="tau for the foreground")
    parser.add_argument("--tau_bg", type=float, default=0.7, help="tau for the background")
    parser.add_argument("--r", type=float, default=0.9, help="the r for the label correction")

    parser.add_argument("--number_epochs", type=int, default=100)

    parser.add_argument("--loss", type=str, default="ce")
    # JSD related
    parser.add_argument("--jsd-lambda", type=float, default=1)
    parser.add_argument("--rho", type=float, default=0.8, help='the threshold when select the target for JSD')

    parser.add_argument("--resize", type=int, default=None)
    parser.add_argument("--ignore-background", action="store_true", default=False)
    parser.add_argument("--background-weight", type=float, default=None)
    parser.add_argument("--cache-dir", type=str, default="")
    parser.add_argument("--balance", action="store_true", default=False)
    args = parser.parse_args()

    parameters = {}
    parameters["data_list"] = args.data_list
    parameters["data_dir"] = args.data_dir
    parameters["save_dir"] = args.save_dir
    parameters["model_name"] = args.model_name
    parameters["seed"] = args.seed
    parameters["batch_size"] = args.batch_size
    parameters["lr"] = args.lr
    parameters["phase"] = {"training":True, "validation":True, "test":True}
    parameters["number_epochs"] = args.number_epochs
    parameters["noise_label"] = args.noise_label
    parameters["noise_level"] = args.noise_level
    parameters["loss"] = args.loss
    # JSD related
    parameters["jsd_lambda"] = args.jsd_lambda
    parameters["rho"] = args.rho

    parameters["resize"] = args.resize
    parameters["cache_dir"] = args.cache_dir
    parameters["label_correction"] = args.label_correction
    parameters["balance"] = args.balance

    # label correction related
    parameters["r"] = args.r
    parameters["conf_threshold"] = args.tau_fg
    parameters["conf_threshold_bg"] = args.tau_bg

    parameters["logic_255"] = args.logic255
    parameters["ignore_background"] = args.ignore_background
    parameters["background_weight"] = args.background_weight
    parameters["no_augmentation"] = True
    parameters["correct_freq"] = args.correct_freq
    run_experiment(parameters)