# ----------------------------------------
# heavily borrowed from Yude Wang, modified by Kangning Liu
# ----------------------------------------

import cv2
import numpy as np
import torch
import random
import PIL
from PIL import Image, ImageOps, ImageFilter
import torch.nn.functional as F
import torchvision.transforms.functional as tf
class RandomCrop(object):
	"""Crop randomly the image in a sample.

	Args:
		output_size (tuple or int): Desired output size. If int, square crop
			is made.
	"""

	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size

	def __call__(self, sample):

		h, w = sample['image'].shape[:2]
		ch = min(h, self.output_size[0])
		cw = min(w, self.output_size[1])
		
		h_space = h - self.output_size[0]
		w_space = w - self.output_size[1]

		if w_space > 0:
			cont_left = 0
			img_left = random.randrange(w_space+1)
		else:
			cont_left = random.randrange(-w_space+1)
			img_left = 0

		if h_space > 0:
			cont_top = 0
			img_top = random.randrange(h_space+1)
		else:
			cont_top = random.randrange(-h_space+1)
			img_top = 0

		key_list = sample.keys()
		for key in key_list:
			if 'image' in key:
				img = sample[key]
				img_crop = np.zeros((self.output_size[0], self.output_size[1], 3), np.float32)
				img_crop[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
						 img[img_top:img_top+ch, img_left:img_left+cw]
				#img_crop = img[img_top:img_top+ch, img_left:img_left+cw]
				sample[key] = img_crop
			elif 'prev_prediction' in key:
				prev_pred = sample[key]
				prev_pred = F.interpolate(prev_pred,  size=(h, w), mode='nearest')
				prev_pred_crop = torch.ones(1, 21, self.output_size[0], self.output_size[1])*np.nan  # tensor
				prev_pred_crop[:,:,cont_top:cont_top+ch, cont_left:cont_left+cw] = \
						 prev_pred[:,:,img_top:img_top+ch, img_left:img_left+cw]
				sample[key] = prev_pred_crop

			elif 'segmentation' == key:
				seg = sample[key]
				seg_crop = np.ones((self.output_size[0], self.output_size[1]), np.float32)*255
				seg_crop[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
						 seg[img_top:img_top+ch, img_left:img_left+cw]
				#seg_crop = seg[img_top:img_top+ch, img_left:img_left+cw]
				sample[key] = seg_crop
			elif 'segmentation2' == key or 'segmentation3' == key or 'segmentationgt' == key:
				seg = sample[key]
				seg_crop = np.ones((self.output_size[0], self.output_size[1]), np.float32)*255
				seg_crop[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
						 seg[img_top:img_top+ch, img_left:img_left+cw]
				#seg_crop = seg[img_top:img_top+ch, img_left:img_left+cw]
				sample[key] = seg_crop
			elif 'segmentation_pseudo' in key:
				seg_pseudo = sample[key]
				seg_crop = np.ones((self.output_size[0], self.output_size[1]), np.float32)*255
				seg_crop[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
						 seg_pseudo[img_top:img_top+ch, img_left:img_left+cw]
				#seg_crop = seg_pseudo[img_top:img_top+ch, img_left:img_left+cw]
				sample[key] = seg_crop

		sample['cropinfo'] = torch.tensor((img_top, ch, img_left, cw, cont_top, cont_left))
		return sample






class CenterCrop(object):
	"""Crop randomly the image in a sample.

	Args:
		output_size (tuple or int): Desired output size. If int, square crop
			is made.
	"""

	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size

	def __call__(self, sample):

		h, w = sample['image'].shape[:2]
		ch = min(h, self.output_size[0])
		cw = min(w, self.output_size[1])
		
		h_space = h - self.output_size[0]
		w_space = w - self.output_size[1]


		if w_space > 0: #cropping is smaller than image
			cont_left = 0
			cont_right = w
			img_left = int(np.ceil(w_space / 2))
			img_right = w - int(np.floor(w_space / 2))
		else:
			cont_left = int(np.ceil(-w_space / 2)) 
			cont_right = self.output_size[1] - int(np.floor(-w_space / 2))
			img_left = 0
			img_right = self.output_size[1]


		if h_space > 0:
			cont_top = 0
			cont_bottom = h
			img_top = int(np.ceil(h_space / 2))
			img_bottom = h - int(np.floor(h_space / 2))
		else:
			cont_top = int(np.ceil(-h_space / 2))
			cont_bottom = self.output_size[0] - int(np.floor(-h_space / 2)) 
			img_top = 0
			img_bottom = self.output_size[0]



		key_list = sample.keys()
		for key in key_list:
			if 'image' in key:
				img = sample[key]
				img_crop = np.zeros((self.output_size[0], self.output_size[1], 3), np.float32)
				img_crop[cont_top:cont_bottom, cont_left:cont_right] = \
						 img[img_top:img_bottom, img_left:img_right]
				#img_crop = img[img_top:img_top+ch, img_left:img_left+cw]
				sample[key] = img_crop
			elif 'prev_prediction' in key:
				prev_pred = sample[key]
				prev_pred = F.interpolate(prev_pred,  size=(h, w), mode='nearest')
				prev_pred_crop = torch.ones(1, 21, self.output_size[0], self.output_size[1])*np.nan  # tensor
				prev_pred_crop[:,:,cont_top:cont_bottom, cont_left:cont_right] = \
						 prev_pred[:,:,img_top:img_bottom, img_left:img_right]
				sample[key] = prev_pred_crop

			elif 'segmentation' == key:
				seg = sample[key]
				seg_crop = np.ones((self.output_size[0], self.output_size[1]), np.float32)*255
				seg_crop[cont_top:cont_bottom, cont_left:cont_right] = \
						 seg[img_top:img_bottom, img_left:img_right]
				#seg_crop = seg[img_top:img_top+ch, img_left:img_left+cw]
				sample[key] = seg_crop
			elif 'segmentation2' == key or 'segmentation3' == key or 'segmentationgt' == key:
				seg = sample[key]
				seg_crop = np.ones((self.output_size[0], self.output_size[1]), np.float32)*255
				seg_crop[cont_top:cont_bottom, cont_left:cont_right] = \
						 seg[img_top:img_bottom, img_left:img_right]
				#seg_crop = seg[img_top:img_top+ch, img_left:img_left+cw]
				sample[key] = seg_crop
			elif 'segmentation_pseudo' in key:
				seg_pseudo = sample[key]
				seg_crop = np.ones((self.output_size[0], self.output_size[1]), np.float32)*255
				seg_crop[cont_top:cont_bottom, cont_left:cont_right] = \
						 seg_pseudo[img_top:img_bottom, img_left:img_right]
				#seg_crop = seg_pseudo[img_top:img_top+ch, img_left:img_left+cw]
				sample[key] = seg_crop

		sample['cropinfo'] = torch.tensor((img_top, ch, img_left, cw, cont_top, cont_left))
		return sample




class RandomHSV(object):
	"""Generate randomly the image in hsv space."""
	def __init__(self, h_r, s_r, v_r):
		self.h_r = h_r
		self.s_r = s_r
		self.v_r = v_r

	def __call__(self, sample):
		image = sample['image']
		hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
		h = hsv[:,:,0].astype(np.int32)
		s = hsv[:,:,1].astype(np.int32)
		v = hsv[:,:,2].astype(np.int32)
		delta_h = random.randint(-self.h_r,self.h_r)
		delta_s = random.randint(-self.s_r,self.s_r)
		delta_v = random.randint(-self.v_r,self.v_r)
		h = (h + delta_h)%180
		s = s + delta_s
		s[s>255] = 255
		s[s<0] = 0
		v = v + delta_v
		v[v>255] = 255
		v[v<0] = 0
		hsv = np.stack([h,s,v], axis=-1).astype(np.uint8)   
		image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.uint8)
		sample['image'] = image
		return sample

class RandomFlip(object):
	"""Randomly flip image"""
	def __init__(self, threshold):
		self.flip_t = threshold
	def __call__(self, sample):
		Flip_sign = False
		if random.random() < self.flip_t:
			Flip_sign =True
			key_list = sample.keys()
			for key in key_list:
				if 'image' in key:
					img = sample[key]
					img = np.flip(img, axis=1)  #h,w,c
					sample[key] = img
				elif 'prev_prediction' in key:
					prev_pred = sample[key]   #1,c,h,w  tensor
					prev_pred = torch.flip(prev_pred, dims=[3])  # flip in the w dimension
					# img_crop = img[img_top:img_top+ch, img_left:img_left+cw]
					sample[key] = prev_pred
				elif 'segmentation' == key:
					seg = sample[key]
					seg = np.flip(seg, axis=1)
					sample[key] = seg
				elif 'segmentation2' == key or 'segmentation3' == key or 'segmentationgt' == key:
					seg = sample[key]
					seg = np.flip(seg, axis=1)
					sample[key] = seg
				elif 'segmentation_pseudo' in key:
					seg_pseudo = sample[key]
					seg_pseudo = np.flip(seg_pseudo, axis=1)
					sample[key] = seg_pseudo
		sample['flipinfo']= Flip_sign
		return sample

class RandomScale(object):
	"""Randomly scale image"""
	def __init__(self, scale_r, is_continuous=False):
		self.scale_r = scale_r
		self.seg_interpolation = cv2.INTER_CUBIC if is_continuous else cv2.INTER_NEAREST

	def __call__(self, sample):
		row, col, _ = sample['image'].shape
		rand_scale = random.random()*(self.scale_r[1] - self.scale_r[0]) + self.scale_r[0]
		key_list = sample.keys()
		for key in key_list:
			if 'image' in key:
				img = sample[key]
				img = cv2.resize(img, None, fx=rand_scale, fy=rand_scale, interpolation=cv2.INTER_CUBIC)
				sample[key] = img
			elif 'prev_prediction' in key:
				prev_pred = sample[key]  # 1,c,h,w
				# prev_pred = F.interpolate(prev_pred, scale_factor=rand_scale, mode='nearest',
				#                              recompute_scale_factor=True)
				prev_pred = F.interpolate(prev_pred, scale_factor=rand_scale, mode='bilinear',align_corners=True,
										  recompute_scale_factor=False)

				# print(prev_pred.size())
				sample[key] = prev_pred
			elif 'segmentation' == key:
				seg = sample[key]
				seg = cv2.resize(seg, None, fx=rand_scale, fy=rand_scale, interpolation=self.seg_interpolation)
				sample[key] = seg
			elif 'segmentation2' == key or 'segmentation3' == key or 'segmentationgt' == key:
				seg = sample[key]
				seg = cv2.resize(seg, None, fx=rand_scale, fy=rand_scale, interpolation=self.seg_interpolation)
				sample[key] = seg
			elif 'segmentation_pseudo' in key:
				seg_pseudo = sample[key]
				seg_pseudo = cv2.resize(seg_pseudo, None, fx=rand_scale, fy=rand_scale, interpolation=self.seg_interpolation)
				sample[key] = seg_pseudo
		sample['scaleinfo'] = rand_scale
		# sample['scaleszie'] = np.array([img.shape[0],img.shape[1]])
		return sample

class RandomBlur(object):
	def __init__(self, scale = 3.3):
		self.scale = scale
	def __call__(self, sample):
		sigma = random.random()
		ksize = int(3.3 * sigma)
		ksize = ksize + 1 if ksize % 2 == 0 else ksize

		key_list = sample.keys()
		for key in key_list:
			if 'image' in key:
				img = sample[key]
				img = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
				sample[key] = img
		# sample['scaleszie'] = np.array([img.shape[0],img.shape[1]])
		return sample



		

class ImageNorm(object):
	"""Randomly scale image"""
	def __init__(self, mean=None, std=None):
		self.mean = mean
		self.std = std
	def __call__(self, sample):
		key_list = sample.keys()
		for key in key_list:
			if 'image' in key:
				image = sample[key].astype(np.float32)
				if self.mean is not None and self.std is not None:
					image[...,0] = (image[...,0]/255 - self.mean[0]) / self.std[0]
					image[...,1] = (image[...,1]/255 - self.mean[1]) / self.std[1]
					image[...,2] = (image[...,2]/255 - self.mean[2]) / self.std[2]
				else:
					image /= 255.0
				sample[key] = image
		return sample

class Multiscale(object):
	def __init__(self, rate_list):
		self.rate_list = rate_list

	def __call__(self, sample):
		image = sample['image']
		row, col, _ = image.shape
		image_multiscale = []
		for rate in self.rate_list:
			rescaled_image = cv2.resize(image, None, fx=rate, fy=rate, interpolation=cv2.INTER_CUBIC)
			sample['image_%f'%rate] = rescaled_image
		return sample


class ToTensor(object):
	"""Convert ndarrays in sample to Tensors."""

	def __call__(self, sample):
		key_list = sample.keys()
		for key in key_list:
			if 'image' in key:
				image = sample[key].astype(np.float32)
				# swap color axis because
				# numpy image: H x W x C
				# torch image: C X H X W
				image = image.transpose((2,0,1))
				sample[key] = torch.from_numpy(image)
				#sample[key] = torch.from_numpy(image.astype(np.float32)/128.0-1.0)
			elif 'prev_prediction' in key:
				prev_pred = sample[key]  # 1,c,h,w
				sample[key] = torch.squeeze(prev_pred,0)   #c,h,w


			elif 'edge' == key:
				edge = sample['edge']
				sample['edge'] = torch.from_numpy(edge.astype(np.float32))
				sample['edge'] = torch.unsqueeze(sample['edge'],0)
			elif 'segmentation' == key:
				segmentation = sample['segmentation']
				sample['segmentation'] = torch.from_numpy(segmentation.astype(np.long))

			elif 'segmentation2' == key or 'segmentation3' == key or 'segmentationgt' == key:
				# segmentation = sample['segmentation2']
				# sample['segmentation2'] = torch.from_numpy(segmentation.astype(np.long))
				segmentation = sample[key]
				sample[key] = torch.from_numpy(segmentation.astype(np.long))

			elif 'segmentation_pseudo' in key:
				segmentation_pseudo = sample[key]
				sample[key] = torch.from_numpy(segmentation_pseudo.astype(np.float32))
			elif 'segmentation_onehot' == key:
				onehot = sample['segmentation_onehot'].transpose((2,0,1))
				sample['segmentation_onehot'] = torch.from_numpy(onehot.astype(np.float32))
			elif 'category' in key:
				sample[key] = torch.from_numpy(sample[key].astype(np.float32))
			elif 'mask' == key:
				mask = sample['mask']
				sample['mask'] = torch.from_numpy(mask.astype(np.float32))
			elif 'feature' == key:
				feature = sample['feature']
				sample['feature'] = torch.from_numpy(feature.astype(np.float32))
		return sample



class AdjustBrightness(object):
	def __init__(self, bf):
		self.bf = bf

	def __call__(self, sample):

		key_list = sample.keys()
		for key in key_list:
			if 'image' in key:
				img = sample[key]
				img = tf.adjust_brightness(img, random.uniform(1 - self.bf, 1 + self.bf))
				sample[key] = img


class AdjustContrast(object):
	def __init__(self, cf):
		self.cf = cf

	def __call__(self, sample):

		key_list = sample.keys()
		for key in key_list:
			if 'image' in key:
				img = sample[key]
				img = tf.adjust_contrast(img, random.uniform(1 - self.cf, 1 + self.cf))
				sample[key] = img


class AdjustSaturation(object):
	def __init__(self, saturation):
		self.saturation = saturation


	def __call__(self, sample):

		key_list = sample.keys()
		for key in key_list:
			if 'image' in key:
				img = sample[key]
				img = tf.adjust_saturation(img, random.uniform(1 - self.saturation, 1 + self.saturation))
				sample[key] = img




class AdjustHue(object):
	def __init__(self, hue):
		self.hue = hue

	def __call__(self, sample):

		key_list = sample.keys()
		for key in key_list:
			if 'image' in key:
				img = sample[key]
				img = tf.adjust_hue(img, random.uniform(-self.hue, self.hue))
				sample[key] = img



class AdjustGamma(object):
	def __init__(self, gamma):
		self.gamma = gamma


	def __call__(self, sample):

		key_list = sample.keys()
		for key in key_list:
			if 'image' in key:
				img = sample[key]
				img = tf.adjust_gamma(img, random.uniform(1, 1 + self.gamma))
				sample[key] = img
