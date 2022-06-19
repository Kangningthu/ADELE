import os, torch, cv2, pickle, copy, random
import numpy as np
from scipy import misc
from PIL import Image
from torch.utils.data import Dataset, Sampler
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


def final_noise_function(mat):
    mode = np.random.choice(["under", "over"])
    iterations = np.random.choice(np.arange(2,5))
    return under_over_seg(mat, iterations, mode)


def under_over_seg(mat, iteration=1, mode="under"):
    target_num = 1000
    mat = np.copy(mat)
    kernel = np.ones((3,3),np.uint8)
    for cls in [1,3,4,2]:
        binary_mat = mat==cls
        foreground_num = np.sum(binary_mat)
        if foreground_num != 0:
            # resize the image to match the foreground pixel number
            h, w = mat.shape
            ratio = np.sqrt(target_num/foreground_num)
            h_new = int(round( h * ratio))
            w_new = int(round( w * ratio))
            resized_img = cv2.resize(binary_mat.astype("uint8"), (w_new, h_new), interpolation=cv2.INTER_CUBIC) > 0
            # erosion or dilation
            if mode == "under":
                binary_mat_processed = cv2.erode(resized_img.astype("uint8"),kernel, iterations =iteration)
            elif mode == "over":
                binary_mat_processed = cv2.dilate(resized_img.astype("uint8"), kernel, iterations=iteration)
            # resize back to the original size
            binary_mat_processed_resized = cv2.resize(binary_mat_processed, (w, h), interpolation=cv2.INTER_CUBIC) > 0
            # fill in the gap
            if mode == "under":
                mat = np.where(binary_mat_processed_resized!=binary_mat, np.zeros(mat.shape), mat)
            elif mode == "over":
                mat = np.where(binary_mat_processed_resized & (mat==0), np.ones(mat.shape)*cls, mat)
    return mat

def under_seg(mat):
    mat = np.copy(mat)
    kernel_small = np.ones((2,2),np.uint8)
    kernel_medium = np.ones((3,3),np.uint8)
    kernel_large = np.ones((5,5),np.uint8)
    for cls in [1,2,3,4]:
        binary_mat = mat==cls
        if cls in [1,3]:
            kernel_used = kernel_small
            iteration = 1
        elif cls == 2:
            kernel_used = kernel_large
            iteration = 2
        else:
            kernel_used = kernel_medium
            iteration = 2
        binary_mat_eroded = cv2.erode(binary_mat.astype("uint8"),kernel_used,iterations =iteration)
        mat = np.where(binary_mat_eroded!=binary_mat, np.zeros(mat.shape), mat)
    return mat

def over_seg(mat):
    mat = np.copy(mat)
    kernel_small = np.ones((2,2),np.uint8)
    kernel_medium = np.ones((3,3),np.uint8)
    kernel_large = np.ones((5,5),np.uint8)
    for cls in [1,2,3,4]:
        if cls in [1,3]:
            kernel_used = kernel_small
        elif cls == 3:
            kernel_used = kernel_large
        else:
            kernel_used = kernel_medium
        binary_mat = mat==cls
        binary_mat_dilated = cv2.dilate(binary_mat.astype("uint8"), kernel_used, iterations=2)
        mat = np.where(binary_mat_dilated, np.ones(mat.shape)*cls, mat)
    return mat

def wrong_seg(mat):
    mat_cp = np.copy(mat)
    channel_0 = np.random.choice([1,2])
    channel_1 = np.random.choice([0,2])
    channel_2 = np.random.choice([0,1])
    mat_cp[0,:,:] = mat[channel_0,:,:]
    mat_cp[1,:,:] = mat[channel_1,:,:]
    mat_cp[2,:,:] = mat[channel_2,:,:]
    return mat_cp

def noise_seg(mat, noise_level=0.05):
    """
    P(out=0 | in=0) = 1-noise_level
    P(out=1234 | in=0) = noise_level/4
    P(out=0 | in=1234) = noise_level
    P(out=1234 | in=1234) = 1-noise_level
    """
    mat = np.copy(mat)
    fate = np.random.uniform(low=0, high=1, size=mat.shape)
    # deal with 0
    is_zero_indicator = mat == 0
    background_flip_to = np.random.choice([1,2,3,4], size=mat.shape)
    mat = np.where( (fate <= noise_level) & is_zero_indicator, background_flip_to, mat)
    # deal with 1,2,3,4
    mat = np.where( (fate <= noise_level) & (~is_zero_indicator), np.zeros(mat.shape), mat)
    return mat

def mixed_seg(mat):
    fate = np.random.uniform(0,1)
    if fate < 0.33:
        return under_seg(mat)
    elif fate < 0.67:
        return over_seg(mat)
    else:
        return noise_seg(mat)


NOISE_LABEL_DICT = {"under":under_seg, "over":over_seg, "wrong":wrong_seg, "noise":noise_seg,
                    "mixed":mixed_seg, "final":final_noise_function}

class StackedRandomAffine(transforms.RandomAffine):
    def __call__(self, imgs):
        """
            img (PIL Image): Image to be transformed.
        Returns:
            PIL Image: Affine transformed image.
        """
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, imgs[0].size)
        return [F.affine(x, *ret, resample=self.resample, fillcolor=self.fillcolor) for x in imgs]


def standarize(img):
    return (img - img.mean()) / img.std()

class BaseDataset(Dataset):
    def __init__(self, parameters, data_list, augmentation=False, noise_label=None, noise_level=None, cache_dir=None):
        self.data_list = data_list
        self.data_dir = parameters["data_dir"]
        self.img_dir = os.path.join(self.data_dir, "img")
        self.seg_dir = os.path.join(self.data_dir, "label")

        # reset seeds
        random.seed(parameters["seed"])
        torch.manual_seed(parameters["seed"])
        torch.cuda.manual_seed(parameters["seed"])
        np.random.seed(parameters["seed"])

        # load cached images and labels if necessary
        if cache_dir is None:
            self.cache_label = None
            self.cache_img = None
        else:
            with open(cache_dir, "rb") as f:
                self.cache_img, self.cache_label = pickle.load(f)
                self.cache_clean_label = copy.deepcopy(self.cache_label)

        # noise label functions
        self.noise_function = None if noise_label is None else NOISE_LABEL_DICT[noise_label]
        if self.noise_function is not None and noise_level is not None:
            noise_number = int(round(noise_level * len(self.data_list)))
            self.noise_index_list = np.random.permutation(np.arange(len(self.data_list)))[:noise_number]
            # add noise to the cached labels
            for i in range(len(self.data_list)):
                if i in self.noise_index_list:
                    img_name = self.data_list[i]
                    self.cache_label[img_name] = self.noise_function(self.cache_label[img_name])
            self.cache_noisy_label = copy.deepcopy(self.cache_label)
        else:
            self.cache_noisy_label = self.cache_clean_label

        # augmentation setting
        self.augmentation = augmentation
        self.augmentation_function = StackedRandomAffine(degrees=(-45, 45), translate=(0.1, 0.1), scale=(0.8, 1.5))

        # transformation setting
        transform_list = []
        if parameters["resize"] is not None:
            transform_list.append(transforms.Resize(size=(parameters["resize"], parameters["resize"]),
                                                     interpolation=0))
        transform_list.append(transforms.ToTensor())
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.data_list)



class BraTSDataset(BaseDataset):
    def __init__(self, parameters, data_list, augmentation=False, noise_label=None):
        super(BraTSDataset, self).__init__(parameters, data_list, augmentation, noise_label)

    def __getitem__(self, index):
        img_name = self.data_list[index]

        # put up paths
        img_path = os.path.join(self.img_dir, img_name)
        seg_path = os.path.join(self.seg_dir, img_name)

        # load images and seg
        img = np.load(img_path).astype("int16")
        seg = np.load(seg_path).astype("int8")
        if self.noise_function is not None:
            seg = self.noise_function(seg)

        # convert to pil image
        img_channel_pils = [Image.fromarray(img[i,:,:].astype("int16")) for i in range(img.shape[0])]
        seg_channel_pils = [Image.fromarray(seg[i,:,:].astype("int8")) for i in range(seg.shape[0])]

        # augmentation
        if self.augmentation:
            aug_res = self.augmentation_function(img_channel_pils + seg_channel_pils)
            img_channel_pils = aug_res[:4]
            seg_channel_pils = aug_res[4:]

        # post-process
        img_channel_torch = [standarize(self.to_tensor(x).float()) for x in img_channel_pils]
        label_channel_torch = [self.to_tensor(x) for x in seg_channel_pils]
        img_torch = torch.cat(img_channel_torch, dim=0)
        label_torch = torch.cat(label_channel_torch, dim=0)
        label_torch[label_torch > 0] = 1

        return img_torch.float(), label_torch.long(), img_name

class SegTHORDataset(BaseDataset):
    def __init__(self, parameters, data_list, augmentation=False, noise_label=None, noise_level=None, cache_dir=None):
        super(SegTHORDataset, self).__init__(parameters, data_list, augmentation, noise_label, noise_level, cache_dir)

    def reset_labels(self, new_labels):
        self.cache_label = new_labels

    def __getitem__(self, index):
        img_name = self.data_list[index]
        # load image and the segmentation label
        if self.cache_img is None:
            img_path = os.path.join(self.img_dir, img_name)
            img = np.load(img_path).astype("int16")
            img -= img.min()
        else:
            img = self.cache_img[img_name]
        if self.cache_label is None:
            seg_path = os.path.join(self.seg_dir, img_name)
            seg = np.load(seg_path).astype("int8")
            # add noise to the label if needed
            if self.noise_function is not None and index in self.noise_index_list:
                seg = self.noise_function(seg)
        else:
            seg = self.cache_label[img_name]
            clean_seg = self.cache_clean_label[img_name]
            original_noisy_seg = self.cache_noisy_label[img_name]

        # convert to pil image
        img_pils = Image.fromarray(img)
        seg_pils = Image.fromarray(seg)
        clean_seg_pils = Image.fromarray(clean_seg)
        original_noisy_seg_pils = Image.fromarray(original_noisy_seg)

        # augmentation
        if self.augmentation:
            img_pils, seg_pils, clean_seg_pils, original_noisy_seg_pils = self.augmentation_function([img_pils, seg_pils, clean_seg_pils, original_noisy_seg_pils])

        # post-process
        img_torch = standarize(self.transform(img_pils).float())
        label_torch = self.transform(seg_pils)
        clean_label_torch = self.transform(clean_seg_pils)
        original_noisy_torch = self.transform(original_noisy_seg_pils)
        return img_torch.float(), label_torch.long(), original_noisy_torch.long(), clean_label_torch.long(), img_name