# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

from __future__ import print_function, division
import os
import torch
import pandas as pd
import cv2
import multiprocessing
from skimage import io
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from datasets.transform import *
from utils.imutils import *
from utils.registry import DATASETS
from datasets.BaseDataset import BaseDataset
from utils.iou_computation import update_iou_stat, compute_iou, get_mask, update_fraction_instancewise, update_fraction_pixelwise

@DATASETS.register_module
class VOCDataset(BaseDataset):
	def __init__(self, cfg, period, transform='none'):
		super(VOCDataset, self).__init__(cfg, period, transform)
		self.dataset_name = 'VOC%d'%cfg.DATA_YEAR
		self.root_dir = os.path.join(cfg.ROOT_DIR,'data','VOCdevkit')
		self.dataset_dir = os.path.join(self.root_dir,self.dataset_name)
		self.rst_dir = os.path.join(self.root_dir,'results',self.dataset_name,'Segmentation')
		self.eval_dir = os.path.join(self.root_dir,'eval_result',self.dataset_name,'Segmentation')
		self.img_dir = os.path.join(self.dataset_dir, 'JPEGImages')
		# print(self.img_dir)
		self.ann_dir = os.path.join(self.dataset_dir, 'Annotations')
		self.seg_dir = os.path.join(self.dataset_dir, 'SegmentationClass')
		self.set_dir = os.path.join(self.dataset_dir, 'ImageSets', 'Segmentation')
		if cfg.DATA_PSEUDO_GT:
			self.pseudo_gt_dir = cfg.DATA_PSEUDO_GT
		else:
			self.pseudo_gt_dir = os.path.join(self.root_dir,'pseudo_gt',self.dataset_name,'Segmentation')

		file_name = None
		if cfg.DATA_AUG and 'train' in self.period:
			file_name = self.set_dir+'/'+period+'aug.txt'
		else:
			file_name = self.set_dir+'/'+period+'.txt'
		df = pd.read_csv(file_name, names=['filename'])
		self.name_list = df['filename'].values
		# print(self.name_list[1])
		if self.dataset_name == 'VOC2012':
			self.categories = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow',
							  'diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
			self.coco2voc = [[0],[5],[2],[16],[9],[44],[6],[3],[17],[62],
							 [21],[67],[18],[19],[4],[1],[64],[20],[63],[7],[72]]

			self.num_categories = len(self.categories)+1
			self.cmap = self.__colormap(len(self.categories)+1)

	def __len__(self):
		return len(self.name_list)

	def load_name(self, idx):
		name = self.name_list[idx]
		return name

	def load_image(self, idx):
		name = self.name_list[idx]
		img_file = self.img_dir + '/' + name + '.jpg'
		image = cv2.imread(img_file)
		image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		return image_rgb

	def load_segmentation(self, idx):
		name = self.name_list[idx]
		seg_file = self.seg_dir + '/' + name + '.png'
		segmentation = np.array(Image.open(seg_file))
		return segmentation

	def load_pseudo_segmentation(self, idx):
		name = self.name_list[idx]
		seg_file = self.pseudo_gt_dir + '/' + name + '.png'
		segmentation = np.array(Image.open(seg_file))
		return segmentation

	def __colormap(self, N):
		"""Get the map from label index to color

		Args:
			N: number of class

			return: a Nx3 matrix

		"""
		cmap = np.zeros((N, 3), dtype = np.uint8)

		def uint82bin(n, count=8):
			"""returns the binary of integer n, count refers to amount of bits"""
			return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

		for i in range(N):
			r = 0
			g = 0
			b = 0
			idx = i
			for j in range(7):
				str_id = uint82bin(idx)
				r = r ^ ( np.uint8(str_id[-1]) << (7-j))
				g = g ^ ( np.uint8(str_id[-2]) << (7-j))
				b = b ^ ( np.uint8(str_id[-3]) << (7-j))
				idx = idx >> 3
			cmap[i, 0] = r
			cmap[i, 1] = g
			cmap[i, 2] = b
		return cmap

	def load_ranked_namelist(self):
		df = self.read_rank_result()
		self.name_list = df['filename'].values

	def label2colormap(self, label):
		m = label.astype(np.uint8)
		r,c = m.shape
		cmap = np.zeros((r,c,3), dtype=np.uint8)
		cmap[:,:,0] = (m&1)<<7 | (m&8)<<3
		cmap[:,:,1] = (m&2)<<6 | (m&16)<<2
		cmap[:,:,2] = (m&4)<<5
		cmap[m==255] = [255,255,255]
		return cmap
	
	def save_result(self, result_list, model_id):
		"""Save test results

		Args:
			result_list(list of dict): [{'name':name1, 'predict':predict_seg1},{...},...]

		"""
		folder_path = os.path.join(self.rst_dir,'%s_%s'%(model_id,self.period))
		if not os.path.exists(folder_path):
			os.makedirs(folder_path)
			
		for sample in result_list:
			file_path = os.path.join(folder_path, '%s.png'%sample['name'])
			cv2.imwrite(file_path, sample['predict'])

	def save_pseudo_gt(self, result_list, folder_path=None):
		"""Save pseudo gt

		Args:
			result_list(list of dict): [{'name':name1, 'predict':predict_seg1},{...},...]

		"""
		i = 1
		folder_path = self.pseudo_gt_dir if folder_path is None else folder_path
		if not os.path.exists(folder_path):
			os.makedirs(folder_path)
		for sample in result_list:
			file_path = os.path.join(folder_path, '%s.png'%(sample['name']))
			cv2.imwrite(file_path, sample['predict'])
			i+=1

	def do_matlab_eval(self, model_id):
		import subprocess
		path = os.path.join(self.root_dir, 'VOCcode')
		eval_filename = os.path.join(self.eval_dir,'%s_result.mat'%model_id)
		cmd = 'cd {} && '.format(path)
		cmd += 'matlab -nodisplay -nodesktop '
		cmd += '-r "dbstop if error; VOCinit; '
		cmd += 'VOCevalseg(VOCopts,\'{:s}\');'.format(model_id)
		cmd += 'accuracies,avacc,conf,rawcounts = VOCevalseg(VOCopts,\'{:s}\'); '.format(model_id)
		cmd += 'save(\'{:s}\',\'accuracies\',\'avacc\',\'conf\',\'rawcounts\'); '.format(eval_filename)
		cmd += 'quit;"'

		print('start subprocess for matlab evaluation...')
		print(cmd)
		subprocess.call(cmd, shell=True)
	
	def do_python_eval(self, model_id):
		predict_folder = os.path.join(self.rst_dir,'%s_%s'%(model_id,self.period))
		gt_folder = self.seg_dir
		TP = []
		P = []
		T = []
		for i in range(self.num_categories):
			TP.append(multiprocessing.Value('i', 0, lock=True))
			P.append(multiprocessing.Value('i', 0, lock=True))
			T.append(multiprocessing.Value('i', 0, lock=True))
		
		def compare(start,step,TP,P,T):
			for idx in range(start,len(self.name_list),step):
				#print('%d/%d'%(idx,len(self.name_list)))
				name = self.name_list[idx]
				predict_file = os.path.join(predict_folder,'%s.png'%name)
				gt_file = os.path.join(gt_folder,'%s.png'%name)
				predict = np.array(Image.open(predict_file)) #cv2.imread(predict_file)
				gt = np.array(Image.open(gt_file))
				cal = gt<255
				mask = (predict==gt) * cal
		  
				for i in range(self.num_categories):
					P[i].acquire()
					P[i].value += np.sum((predict==i)*cal)
					P[i].release()
					T[i].acquire()
					T[i].value += np.sum((gt==i)*cal)
					T[i].release()
					TP[i].acquire()
					TP[i].value += np.sum((gt==i)*mask)
					TP[i].release()
		p_list = []
		for i in range(8):
			p = multiprocessing.Process(target=compare, args=(i,8,TP,P,T))
			p.start()
			p_list.append(p)
		for p in p_list:
			p.join()
		IoU = []
		for i in range(self.num_categories):
			IoU.append(TP[i].value/(T[i].value+P[i].value-TP[i].value+1e-10))
		loglist = {}
		for i in range(self.num_categories):
			if i == 0:
				print('%11s:%7.3f%%'%('background',IoU[i]*100),end='\t')
				loglist['background'] = IoU[i] * 100 
			else:
				if i%2 != 1:
					print('%11s:%7.3f%%'%(self.categories[i-1],IoU[i]*100),end='\t')
				else:
					print('%11s:%7.3f%%'%(self.categories[i-1],IoU[i]*100))
				loglist[self.categories[i-1]] = IoU[i] * 100
					
		miou = np.mean(np.array(IoU))
		print('\n======================================================')
		print('%11s:%7.3f%%'%('mIoU',miou*100))
		loglist['mIoU'] = miou * 100
		return loglist

	def __coco2voc(self, m):
		r,c = m.shape
		result = np.zeros((r,c),dtype=np.uint8)
		for i in range(0,21):
			for j in self.coco2voc[i]:
				result[m==j] = i
		return result

	def do_python_eval_pseudo(self, path):
		self.seg_dir_gt = os.path.join(self.dataset_dir, 'SegmentationClassAug')
		# predict_folder = os.path.join(self.rst_dir, '%s_%s' % (model_id, self.period))
		predict_folder = path
		gt_folder = self.seg_dir_gt
		TP = []
		P = []
		T = []
		for i in range(self.num_categories):
			TP.append(multiprocessing.Value('i', 0, lock=True))
			P.append(multiprocessing.Value('i', 0, lock=True))
			T.append(multiprocessing.Value('i', 0, lock=True))

		def compare(start, step, TP, P, T):
			for idx in range(start, len(self.name_list), step):
				# print('%d/%d'%(idx,len(self.name_list)))
				name = self.name_list[idx]
				predict_file = os.path.join(predict_folder, '%s.png' % name)
				gt_file = os.path.join(gt_folder, '%s.png' % name)
				predict = np.array(Image.open(predict_file))  # cv2.imread(predict_file)
				gt = np.array(Image.open(gt_file))
				cal = gt < 255
				mask = (predict == gt) * cal

				for i in range(self.num_categories):
					P[i].acquire()
					P[i].value += np.sum((predict == i) * cal)
					P[i].release()
					T[i].acquire()
					T[i].value += np.sum((gt == i) * cal)
					T[i].release()
					TP[i].acquire()
					TP[i].value += np.sum((gt == i) * mask)
					TP[i].release()

		p_list = []
		for i in range(8):
			p = multiprocessing.Process(target=compare, args=(i, 8, TP, P, T))
			p.start()
			p_list.append(p)
		for p in p_list:
			p.join()
		IoU = []
		for i in range(self.num_categories):
			IoU.append(TP[i].value / (T[i].value + P[i].value - TP[i].value + 1e-10))
		loglist = {}
		for i in range(self.num_categories):
			if i == 0:
				print('%11s:%7.3f%%' % ('background', IoU[i] * 100), end='\t')
				loglist['background'] = IoU[i] * 100
			else:
				if i % 2 != 1:
					print('%11s:%7.3f%%' % (self.categories[i - 1], IoU[i] * 100), end='\t')
				else:
					print('%11s:%7.3f%%' % (self.categories[i - 1], IoU[i] * 100))
				loglist[self.categories[i - 1]] = IoU[i] * 100

		miou = np.mean(np.array(IoU))
		print('\n======================================================')
		print('%11s:%7.3f%%' % ('mIoU', miou * 100))
		loglist['mIoU'] = miou * 100
		return loglist


	def do_python_eval_two_list(self, path1, path2):
		# self.seg_dir_gt = os.path.join(self.dataset_dir, 'SegmentationClassAug')
		# predict_folder = os.path.join(self.rst_dir, '%s_%s' % (model_id, self.period))
		predict_folder = path1
		gt_folder = path2
		TP = []
		P = []
		T = []
		for i in range(self.num_categories):
			TP.append(multiprocessing.Value('i', 0, lock=True))
			P.append(multiprocessing.Value('i', 0, lock=True))
			T.append(multiprocessing.Value('i', 0, lock=True))

		def compare(start, step, TP, P, T):
			for idx in range(start, len(self.name_list), step):
				# print('%d/%d'%(idx,len(self.name_list)))
				name = self.name_list[idx]
				predict_file = os.path.join(predict_folder, '%s.png' % name)
				gt_file = os.path.join(gt_folder, '%s.png' % name)
				predict = np.array(Image.open(predict_file))  # cv2.imread(predict_file)
				gt = np.array(Image.open(gt_file))
				cal = gt < 255
				mask = (predict == gt) * cal

				for i in range(self.num_categories):
					P[i].acquire()
					P[i].value += np.sum((predict == i) * cal)
					P[i].release()
					T[i].acquire()
					T[i].value += np.sum((gt == i) * cal)
					T[i].release()
					TP[i].acquire()
					TP[i].value += np.sum((gt == i) * mask)
					TP[i].release()

		p_list = []
		for i in range(8):
			p = multiprocessing.Process(target=compare, args=(i, 8, TP, P, T))
			p.start()
			p_list.append(p)
		for p in p_list:
			p.join()
		IoU = []
		for i in range(self.num_categories):
			IoU.append(TP[i].value / (T[i].value + P[i].value - TP[i].value + 1e-10))
		loglist = {}
		for i in range(self.num_categories):
			if i == 0:
				print('%11s:%7.3f%%' % ('background', IoU[i] * 100), end='\t')
				loglist['background'] = IoU[i] * 100
			else:
				if i % 2 != 1:
					print('%11s:%7.3f%%' % (self.categories[i - 1], IoU[i] * 100), end='\t')
				else:
					print('%11s:%7.3f%%' % (self.categories[i - 1], IoU[i] * 100))
				loglist[self.categories[i - 1]] = IoU[i] * 100

		miou = np.mean(np.array(IoU))
		print('\n======================================================')
		print('%11s:%7.3f%%' % ('mIoU', miou * 100))
		loglist['mIoU'] = miou * 100
		return loglist


	def do_python_val_clean_wrong_one_process(self, model_id):
		predict_folder = os.path.join(self.rst_dir, '%s_%s' % (model_id, self.period))
		self.seg_dir_gt = os.path.join(self.dataset_dir, 'SegmentationClassAug')
		gt_folder = self.seg_dir_gt

		# stat for each epoch
		TP_clean_epoch = [0] * 21
		P_clean_epoch = [0] * 21
		T_clean_epoch = [0] * 21

		TP_wrong_epoch = [0] * 21
		P_wrong_epoch = [0] * 21
		T_wrong_epoch = [0] * 21

		TP_gt_epoch = [0] * 21
		P_gt_epoch = [0] * 21
		T_gt_epoch = [0] * 21

		loglist_clean = {}
		loglist_wrong = {}
		loglist_gt = {}

		for idx in range(len(self.name_list)):
			# print(idx)
			name = self.name_list[idx]
			gt_file = os.path.join(gt_folder, '%s.png' % name)
			gt_np = np.array(Image.open(gt_file))

			# the label produced by CAM
			seg_file = '/scratch/kl3141/seam/SEAM-master/results/CAM_original_val_aff' + '/' + name + '.png'
			segmentation1 = Image.open(seg_file)
			label_np = np.array(segmentation1)

			r, c = gt_np.shape

			predict_file = os.path.join(predict_folder, '%s.png' % name)
			predict = np.array(Image.open(predict_file))  # cv2.imread(predict_file)  #h,w


			pred_np = predict

			mask_clean = (gt_np == label_np)

			gt_np_clean = (gt_np + 1) * mask_clean - 1
			gt_np_clean[gt_np_clean < 0] = 255

			gt_np_wrong = (gt_np + 1) * (~mask_clean) - 1
			gt_np_wrong[gt_np_wrong < 0] = 255

			label_np_clean = (label_np + 1) * mask_clean - 1
			label_np_clean[label_np_clean < 0] = 255

			label_np_wrong = (label_np + 1) * (~mask_clean) - 1
			label_np_wrong[label_np_wrong < 0] = 255

			TP_clean_epoch, P_clean_epoch, T_clean_epoch = update_iou_stat(pred_np, gt_np_clean, TP_clean_epoch,
																		   P_clean_epoch, T_clean_epoch)
			TP_wrong_epoch, P_wrong_epoch, T_wrong_epoch = update_iou_stat(pred_np, gt_np_wrong, TP_wrong_epoch,
																		   P_wrong_epoch, T_wrong_epoch)
			TP_gt_epoch, P_gt_epoch, T_gt_epoch = update_iou_stat(pred_np, gt_np, TP_gt_epoch, P_gt_epoch, T_gt_epoch)

		IoU_clean_epoch = compute_iou(TP_clean_epoch, P_clean_epoch, T_clean_epoch)
		IoU_wrong_epoch = compute_iou(TP_wrong_epoch, P_wrong_epoch, T_wrong_epoch)
		IoU_gt_epoch = compute_iou(TP_gt_epoch, P_gt_epoch, T_gt_epoch)

		for indx, class_name in enumerate(
				['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
				 'cow',
				 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
				 'tvmonitor']):
			loglist_clean[class_name] = IoU_clean_epoch[indx]
			loglist_wrong[class_name] = IoU_wrong_epoch[indx]
			loglist_gt[class_name] = IoU_gt_epoch[indx]
		mIoU_clean_epoch = np.mean(np.array(IoU_clean_epoch))
		mIoU_wrong_epoch = np.mean(np.array(IoU_wrong_epoch))
		mIoU_gt_epoch = np.mean(np.array(IoU_gt_epoch))
		loglist_clean['mIoU'] = mIoU_clean_epoch
		loglist_wrong['mIoU'] = mIoU_wrong_epoch
		loglist_gt['mIoU'] = mIoU_gt_epoch
		return loglist_clean, loglist_wrong, loglist_gt



	def do_python_memorization_IoU_one_process(self, model_id, num_classes=21):
		predict_folder = os.path.join(self.rst_dir, '%s_%s' % (model_id, self.period))
		self.seg_dir_gt = os.path.join(self.dataset_dir, 'SegmentationClassAug')
		gt_folder = self.seg_dir_gt

		# stat for each epoch
		TP_clean_epoch = [0] * num_classes
		P_clean_epoch = [0] * num_classes
		T_clean_epoch = [0] * num_classes

		TP_wrong_epoch = [0] * num_classes
		P_wrong_epoch = [0] * num_classes
		T_wrong_epoch = [0] * num_classes


		TP_wrong_mem_epoch = [0] * num_classes
		P_wrong_mem_epoch = [0] * num_classes
		T_wrong_mem_epoch = [0] * num_classes

		TP_gt_epoch = [0] * num_classes
		P_gt_epoch = [0] * num_classes
		T_gt_epoch = [0] * num_classes

		loglist_clean = {}
		loglist_wrong = {}
		loglist_wrong_mem = {}
		loglist_gt = {}

		for idx in range(len(self.name_list)):
			# print(idx)
			name = self.name_list[idx]
			gt_file = os.path.join(gt_folder, '%s.png' % name)
			gt_np = np.array(Image.open(gt_file))

			# the label produced by CAM
			if 'train' in self.period:
				seg_file = '/scratch/kl3141/seam/SEAM-master/results/aff_rw_aug' + '/' + name + '.png'
			else:
				seg_file = '/scratch/kl3141/seam/SEAM-master/results/CAM_original_val_aff' + '/' + name + '.png'
			segmentation1 = Image.open(seg_file)
			label_np = np.array(segmentation1)

			r, c = gt_np.shape

			predict_file = os.path.join(predict_folder, '%s.png' % name)
			predict = np.array(Image.open(predict_file))  # cv2.imread(predict_file)  #h,w


			pred_np = predict

			mask_clean = (gt_np == label_np)

			gt_np_clean = (gt_np + 1) * mask_clean - 1
			gt_np_clean[gt_np_clean < 0] = 255

			gt_np_wrong = (gt_np + 1) * (~mask_clean) - 1
			gt_np_wrong[gt_np_wrong < 0] = 255

			label_np_clean = (label_np + 1) * mask_clean - 1
			label_np_clean[label_np_clean < 0] = 255

			label_np_wrong = (label_np + 1) * (~mask_clean) - 1
			label_np_wrong[label_np_wrong < 0] = 255

			TP_clean_epoch, P_clean_epoch, T_clean_epoch = update_iou_stat(pred_np, gt_np_clean, TP_clean_epoch,
																		   P_clean_epoch, T_clean_epoch)
			TP_wrong_epoch, P_wrong_epoch, T_wrong_epoch = update_iou_stat(pred_np, gt_np_wrong, TP_wrong_epoch,
																		   P_wrong_epoch, T_wrong_epoch)
			TP_gt_epoch, P_gt_epoch, T_gt_epoch = update_iou_stat(pred_np, gt_np, TP_gt_epoch, P_gt_epoch, T_gt_epoch)


			TP_wrong_mem_epoch, P_wrong_mem_epoch, T_wrong_mem_epoch = update_iou_stat(pred_np, label_np_wrong, TP_wrong_mem_epoch, P_wrong_mem_epoch, T_wrong_mem_epoch)




		IoU_clean_epoch = compute_iou(TP_clean_epoch, P_clean_epoch, T_clean_epoch)
		IoU_wrong_epoch = compute_iou(TP_wrong_epoch, P_wrong_epoch, T_wrong_epoch)
		IoU_gt_epoch = compute_iou(TP_gt_epoch, P_gt_epoch, T_gt_epoch)
		IoU_wrong_mem_epoch = compute_iou(TP_wrong_mem_epoch, P_wrong_mem_epoch, T_wrong_mem_epoch)

		for indx, class_name in enumerate(
				['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
				 'cow',
				 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
				 'tvmonitor']):
			loglist_clean[class_name] = IoU_clean_epoch[indx]
			loglist_wrong[class_name] = IoU_wrong_epoch[indx]
			loglist_gt[class_name] = IoU_gt_epoch[indx]
			loglist_wrong_mem[class_name] = IoU_wrong_mem_epoch[indx]

		mIoU_clean_epoch = np.mean(np.array(IoU_clean_epoch))
		mIoU_wrong_epoch = np.mean(np.array(IoU_wrong_epoch))
		mIoU_gt_epoch = np.mean(np.array(IoU_gt_epoch))
		mIoU_wrong_mem_epoch = np.mean(np.array(IoU_wrong_mem_epoch))


		loglist_clean['mIoU'] = mIoU_clean_epoch
		loglist_wrong['mIoU'] = mIoU_wrong_epoch
		loglist_gt['mIoU'] = mIoU_gt_epoch
		loglist_wrong_mem['mIoU'] = mIoU_wrong_mem_epoch


		return loglist_clean, loglist_wrong, loglist_gt, loglist_wrong_mem

	def do_python_memorization_fraction_one_process(self, model_id, num_classes = 21, split='train'):
		predict_folder = os.path.join(self.rst_dir, '%s_%s' % (model_id, self.period))
		self.seg_dir_gt = os.path.join(self.dataset_dir, 'SegmentationClassAug')
		gt_folder = self.seg_dir_gt

		# stat for each epoch


		# wrong_correct = [0]* num_classes
		# wrong_memorized = [0]*num_classes
		# wrong_others = [0] * num_classes
		# clean_correct = [0] * num_classes
		# clean_incorrect = [0] * num_classes

		mem_item_list_instancewise = np.zeros((5, num_classes))
		mem_item_list_pixelwise = np.zeros((5, num_classes, 2))   # 5 items, 21 classes, (sum_target_pixel, total_pixel_sum)


		for idx in range(len(self.name_list)):
			# print(idx)
			name = self.name_list[idx]
			gt_file = os.path.join(gt_folder, '%s.png' % name)
			gt_np = np.array(Image.open(gt_file))

			# the label produced by CAM
			if 'train' in self.period:
				seg_file = '/scratch/kl3141/seam/SEAM-master/results/aff_rw_aug' + '/' + name + '.png'
			else:
				seg_file = '/scratch/kl3141/seam/SEAM-master/results/CAM_original_val_aff' + '/' + name + '.png'
			segmentation1 = Image.open(seg_file)
			label_np = np.array(segmentation1)

			r, c = gt_np.shape

			predict_file = os.path.join(predict_folder, '%s.png' % name)
			predict = np.array(Image.open(predict_file))  # cv2.imread(predict_file)  #h,w

			pred_np = predict

			mask_list = get_mask(gt_np, label_np, pred_np)  # wrong_mask_correct,wrong_mask_memorized,wrong_mask_others,clean_mask_correct,clean_mask_incorrect

			for item_indx in range(5):
				# mem_item_list_instancewise[item_indx] = update_fraction_instancewise(mask_list[item_indx], gt_np, mem_item_list_instancewise[item_indx], num_classes=num_classes)
				mem_item_list_pixelwise[item_indx] = update_fraction_pixelwise(mask_list[item_indx], gt_np, mem_item_list_pixelwise[item_indx],  num_classes=num_classes)

		return mem_item_list_instancewise, mem_item_list_pixelwise