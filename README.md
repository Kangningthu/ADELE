# ADELE (Adaptive Early-Learning Correction for Segmentation from Noisy Annotations) (CVPR 2022 Oral)


Sheng Liu*, Kangning Liu*, Weicheng Zhu, Yiqiu Shen, Carlos Fernandez-Granda

(* The first two authors contribute equally, order decided by coin flipping.)




Official Implementation of [Adaptive Early-Learning Correction for Segmentation from Noisy Annotations](https://arxiv.org/abs/2110.03740) (CVPR 2022 Oral)

## PASCAL VOC dataset
Thanks to the work of Yude Wang, the code of this repository borrow heavily from his SEAM repository, and we follw the same pipeline to verify the effectiveness of our ADELE. 
We use the same ImageNet pretrained ResNet38 model as SEAM, which can be downloaded from https://github.com/YudeWang/semantic-segmentation-codebase/tree/main/experiment/seamv1-pseudovoc






The code related to PASCAL VOC locates in the main folder, we provide the trained model for SEAM+ADELE in the following link

https://drive.google.com/file/d/10cTOraETOmb2jOCJ4E0m_y9lrjrA3g2u/view?usp=sharing

We use two NVIDIA Quadro RTX 8000 GPUs to train the model, if you encounter out of memory issue, please consider decreasing the resolution of the input image.

### Installation
- Install python dependencies.
```
pip install -r requirements.txt
```

Note that we use comet to record the statistics online. Comet is similar to tensorboard, more information can found via https://www.comet.ml/site/ . 
- Create softlink to your dataset. Make sure that the dataset can be accessed by `$your_dataset_path/VOCdevkit/VOC2012...`
```
ln -s $your_dataset_path data
```




Inference code is the same as the official code for SEAM. Attach the Code link provide by the SEAM author: https://github.com/YudeWang/semantic-segmentation-codebase/tree/main/experiment/seamv1-pseudovoc




For the training code, an example script for ADELE would be:

```
python train.py \
  --EXP_NAME EXP_name \
  --Lambda1 1 --TRAIN_BATCHES 10 --TRAIN_LR 0.001 --mask_threshold 0.8 \
   --scale_index 0 --flip yes --CRF yes  \
   --dict_save_scale_factor 1  --npl_metrics 0 \
   --api_key API_key \
   --r_threshold 0.9 --Reinit_dict yes \
  --DATA_PSEUDO_GT Inital_Pseudo_Label_Location
```



We store some default value for the arguments in the config.py file, those value would be passed to arguments as cfg.XXX. You may change the default value in the config.py or change that via arguments in the script.
It is especially important to assign the path of your initial pseudo annotation via --DATA_PSEUDO_GT or specify that in cfg.DATA_PSEUDO_GT in the config.py file. For the detailed method to obtain the initial pseudo annotation, please refer to the related method such as AffinityNet, SEAM, ICD, NSROM, etc. 


The arguments represent:

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
                        help="threshold to select the mask for Consistency loss computation ")
    parser.add_argument('--DATA_WORKERS', type=int, default=cfg.DATA_WORKERS,
                        help="number of workers in dataloader")


    parser.add_argument('--mask_threshold', type=float, default=0.8,
                        help="only the region with high probability and disagree with Pseudo label be updated")
    parser.add_argument('--TRAIN_LR', type=float,
                        default=cfg.TRAIN_LR,
                        help="the path of trained weight")
    parser.add_argument('--TRAIN_ITERATION', type=int,
                        default=cfg.TRAIN_ITERATION,
                        help="the training iteration number")
    parser.add_argument('--DATA_RANDOMCROP', type=int, default=cfg.DATA_RANDOMCROP,
                        help="the resolution of random crop")



    # related to the pseudo label updating
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
                        help="The api_key of Comet, please refer to https://www.comet.ml/site/ for more information"
    parser.add_argument('--online', type=str2bool, nargs='?',
                        const=True, default=True,
                        help="False when use Comet offline")








## SegTHOR dataset
The code related to SegTHOR locates in the folder SegThor, please go to the subdirectory SegThor
### Installation

- Install python dependencies.
```
pip install -r requirements.txt
```
- Downlaod the SegTHOR dataset and conduct data preprocessing, resize all the image to the size of 256*256 using linear interpolation of opencv_python (INTER_LINEAR). 

The details of public SegTHOR dataset can be found in [this link](https://competitions.codalab.org/competitions/21145).

In this study, we randomly assign patients in the original training set into training, validation, and test set using following scheme:

- training set: ['Patient_01', 'Patient_02', 'Patient_03', 'Patient_04',
       'Patient_05', 'Patient_06', 'Patient_07', 'Patient_09',
       'Patient_10', 'Patient_11', 'Patient_12', 'Patient_13',
       'Patient_14', 'Patient_15', 'Patient_16', 'Patient_17',
       'Patient_18', 'Patient_19', 'Patient_20', 'Patient_22',
       'Patient_24', 'Patient_25', 'Patient_26', 'Patient_28',
       'Patient_30', 'Patient_31', 'Patient_33', 'Patient_36',
       'Patient_38', 'Patient_39', 'Patient_40']
- validation set: ['Patient_21', 'Patient_23', 'Patient_27', 'Patient_29',
       'Patient_37']
- test set: ['Patient_08', 'Patient_27', 'Patient_32', 'Patient_34',
       'Patient_35']
       
We used only slices that contain foreground class and downsampled all slices into 256 * 256 pixels using linear interpolation.

### Experiments
Here is the example script of ADELE:
```
python3  brat/train_segthor.py \
 --cache-dir DIR_OF_THE_DATA   \
 --data-list  DIR_OF_THE_DATALIST \
 --save-dir  MODEL_SAVE_DIR \
 --model-name MODEL_NAME \
 --seed 0 \
 --jsd-lambda 1 \
 --rho 0.8 \
 --label-correction \
 --tau_fg 0.7 \
 --tau_bg 0.7 \
 --r 0.9
```

where the arguments represent:
* `cache-dir` - Parent dir of the datalist, tr.pkl, val.pkl, ts.pkl, which are the input data for training, validation and testing set.
* `data-list` - Parent dir of the data_list.pkl file, which is the list of names for the input data.
* `save-dir` - Folder, where models and results will be saved.
* `model-name` - Name of the model.
* `seed` - the random seed of the noise realization, default 0.
* `jsd-lambda` - the consistency strength, if set to 0, no consistency regularization will be applied, default 1. 
* `rho` - consistency confidence threshold, this is the threshold on the confidence of model's prediction to decide which examples are applied with consistency regularization
* `label-correction` - whether to conduct label correction, if set this arguments, the model will do label correction, default False.
* `tau_fg, tau_bg` - label correction confidence threshold for foreground and background, in the main paper and all the experiment, we set these two values to be the same for simplicity, default 0.7. 
*  `r` - curve fitting threshold to control when a specific semantic category will be corrected, default 0.9. 



Here is the example script of baseline:
```
python3  brat/train_segthor.py \
 --cache-dir DIR_OF_THE_DATA   \
 --data-list  DIR_OF_THE_DATALIST \
 --save-dir  MODEL_SAVE_DIR \
 --model-name MODEL_NAME \
 --seed 0 \
 --jsd-lambda 0 
```








##Citation

Please cite our paper if the code is helpful to your research.
```
@article{liu2021adaptive,
  title={Adaptive Early-Learning Correction for Segmentation from Noisy Annotations},
  author={Liu, Sheng and Liu, Kangning and Zhu, Weicheng and Shen, Yiqiu and Fernandez-Granda, Carlos},
  journal={CVPR 2022},
  year={2022}
}
```

