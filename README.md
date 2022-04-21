# ADELE (Adaptive Early-Learning Correction for Segmentation from Noisy Annotations)


Sheng Liu*, Kangning Liu*, Weicheng Zhu, Yiqiu Shen, Carlos Fernandez-Granda

(* The first two authors contribute equally, order decided by coin flipping.)




Official Implementation of [Adaptive Early-Learning Correction for Segmentation from Noisy Annotations](https://arxiv.org/abs/2110.03740) (CVPR 2022 Oral)


## SegTHOR dataset

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



## PASCAL VOC dataset

we provide the trained model in the following link

https://drive.google.com/file/d/10cTOraETOmb2jOCJ4E0m_y9lrjrA3g2u/view?usp=sharing

The training code will be released soon, inference code is the same as the official code for SEAM. Attach the Code link provide by the SEAM author: https://github.com/YudeWang/semantic-segmentation-codebase/tree/main/experiment/seamv1-pseudovoc

```
@article{liu2021adaptive,
  title={Adaptive Early-Learning Correction for Segmentation from Noisy Annotations},
  author={Liu, Sheng and Liu, Kangning and Zhu, Weicheng and Shen, Yiqiu and Fernandez-Granda, Carlos},
  journal={CVPR 2022},
  year={2022}
}
```

