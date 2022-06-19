import numpy as np
import torch
import torch.nn.functional as F
import cv2
from utils.DenseCRF import *
#from cv2.ximgproc import l0Smooth

def color_pro(pro, img=None, mode='hwc'):
	H, W = pro.shape
	pro_255 = (pro*255).astype(np.uint8)
	pro_255 = np.expand_dims(pro_255,axis=2)
	color = cv2.applyColorMap(pro_255,cv2.COLORMAP_JET)
	color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
	if img is not None:
		rate = 0.5
		if mode == 'hwc':
			assert img.shape[0] == H and img.shape[1] == W
			color = cv2.addWeighted(img,rate,color,1-rate,0)
		elif mode == 'chw':
			assert img.shape[1] == H and img.shape[2] == W
			img = np.transpose(img,(1,2,0))
			color = cv2.addWeighted(img,rate,color,1-rate,0)
			color = np.transpose(color,(2,0,1))
	else:
		if mode == 'chw':
			color = np.transpose(color,(2,0,1))
	return color

def generate_vis(p, gt, img, func_label2color, threshold=0.1, norm=True, crf=False):
	# All the input should be numpy.array
	# img should be 0-255 uint8
	C, H, W = p.shape

	if norm:
		prob = max_norm(p, 'numpy')
	else:
		prob = p
	if gt is not None:
		prob = prob * gt
	prob[prob<=0] = 1e-5
	if threshold is not None:
		prob[0,:,:] = np.power(1-np.max(prob[1:,:,:],axis=0,keepdims=True), 4)

	CLS = ColorCLS(prob, func_label2color)
	CAM = ColorCAM(prob, img)
	if crf:
		prob_crf = dense_crf(prob, img, n_classes=C, n_iters=1)
		CLS_crf = ColorCLS(prob_crf, func_label2color)
		CAM_crf = ColorCAM(prob_crf, img)
		return CLS, CAM, CLS_crf, CAM_crf
	else:
		return CLS, CAM

def max_norm(p, version='torch', e=1e-5):
	if version is 'torch':
		if p.dim() == 3:
			C, H, W = p.size()
			p = F.relu(p, inplace=True)
			max_v = torch.max(p.view(C,-1),dim=-1)[0].view(C,1,1)
			min_v = torch.min(p.view(C,-1),dim=-1)[0].view(C,1,1)
			p = F.relu(p-min_v-e, inplace=True)/(max_v-min_v+e)
		elif p.dim() == 4:
			N, C, H, W = p.size()
			p = F.relu(p, inplace=True)
			max_v = torch.max(p.view(N,C,-1),dim=-1)[0].view(N,C,1,1)
			min_v = torch.min(p.view(N,C,-1),dim=-1)[0].view(N,C,1,1)
			p = F.relu(p-min_v-e, inplace=True)/(max_v-min_v+e)
	elif version is 'numpy' or version is 'np':
		if p.ndim == 3:
			C, H, W = p.shape
			p[p<e] = 0
			max_v = np.max(p,(1,2),keepdims=True)
			min_v = np.min(p,(1,2),keepdims=True)
			p = (p-min_v)/(max_v-min_v+e)
		elif p.ndim == 4:
			N, C, H, W = p.shape
			p[p<e] = 0
			max_v = np.max(p,(2,3),keepdims=True)
			min_v = np.min(p,(2,3),keepdims=True)
			p = (p-min_v)/(max_v-min_v+e)
	return p

def ColorCAM(prob, img):
	assert prob.ndim == 3
	C, H, W = prob.shape
	colorlist = []
	for i in range(C):
		colorlist.append(color_pro(prob[i,:,:],img=img,mode='chw'))
	CAM = np.array(colorlist)/255.0
	return CAM

def ColorCLS(prob, func_label2color):
	assert prob.ndim == 3
	prob_idx = np.argmax(prob, axis=0)
	CLS = func_label2color(prob_idx).transpose((2,0,1))
	return CLS


def num_unique_class_per_img_wo_augarea(pred, mask):
	"""
	how many unique classes are there per prediction (the argmax classes across spatial location), we also choose to maskout the aug area
	:param pred: b,c,h,w
	:param mask: b,h,w  (0 or 1 in each element) do not want to compute the region in the augment area, make it irrelevant
	:return:
	"""
	b, c, h, w = pred.size()
	argmax_probs = ((torch.argmax(pred, dim=1, keepdim=False)+1)*mask).view(b,-1) # b, h*w
	unique_ele_mixlabel = [len(torch.unique(argmax_probs[i,:]))-1 for i in range(b)]  # b
	mean_un_ele_mixlabel = np.mean(unique_ele_mixlabel)
	return mean_un_ele_mixlabel


