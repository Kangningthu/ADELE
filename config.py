# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------
import torch
import argparse
import os
import sys
import cv2
import time

config_dict = {
		'EXP_NAME': 'Experiment',
		'GPUS': 2,
		'TEST_GPUS': 1,
		'DATA_NAME': 'VOCTrainwsegDataset',
		'DATA_YEAR': 2012,
		'DATA_AUG': True,
		'DATA_WORKERS': 8,
		'DATA_MEAN': [0.485, 0.456, 0.406],
		'DATA_STD': [0.229, 0.224, 0.225],
		'DATA_RANDOMCROP': 448,
		'DATA_RANDOMSCALE': [0.5, 1.5],
		'DATA_RANDOM_H': 10,
		'DATA_RANDOM_S': 10,
		'DATA_RANDOM_V': 10,
		'DATA_RANDOMFLIP': 0.5,
		'DATA_PSEUDO_GT': '/scratch/kl3141/seam/SEAM-master/results/aff_rw_aug',
	    'DATA_FEATURE_DIR':False,
		
		'MODEL_NAME': 'deeplabv1',
		'MODEL_BACKBONE': 'resnet38',
		'MODEL_BACKBONE_PRETRAIN': True,
		'MODEL_NUM_CLASSES': 21,
		'MODEL_FREEZEBN': False,

		'TRAIN_LR': 0.001,
		'TRAIN_MOMENTUM': 0.9,
		'TRAIN_WEIGHT_DECAY': 0.0005,
		'TRAIN_BN_MOM': 0.0003,
		'TRAIN_POWER': 0.9,
		'TRAIN_BATCHES': 10,
		'TRAIN_SHUFFLE': True,
		'TRAIN_MINEPOCH': 0,
		'TRAIN_ITERATION': 20000,
		'TRAIN_TBLOG': True,

		'TEST_MULTISCALE': [0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
		'TEST_FLIP': True,
		'TEST_CRF': True,
		'TEST_BATCHES': 1,

		'MODEL_BACKBONE_DILATED':False,
		'MODEL_BACKBONE_MULTIGRID':False,
		'MODEL_BACKBONE_DEEPBASE': False,

		'scale_factor':0.7,
		'scale_factor2': 1.5,
		'lambda_seg': 0.5,

}

config_dict['ROOT_DIR'] = os.path.abspath(os.path.dirname("__file__"))
config_dict['MODEL_SAVE_DIR'] = os.path.join(config_dict['ROOT_DIR'],'model',config_dict['EXP_NAME'])
config_dict['TRAIN_CKPT'] = None
config_dict['LOG_DIR'] = os.path.join(config_dict['ROOT_DIR'],'log',config_dict['EXP_NAME'])
sys.path.insert(0, os.path.join(config_dict['ROOT_DIR'], 'lib'))
