import numpy as np
import cv2
import gc
import matplotlib.pyplot as plt
from os.path import join

from keras.utils import Sequence

class PascalVocGenerator(Sequence):
    
    classes = {
        "background": np.array([0, 0, 0]),
        "aeroplane": np.array([128, 0, 0]),
        "bicycle": np.array([0, 128, 0]),
        "bird": np.array([128, 128, 0]),
        "boat": np.array([0, 0, 128]),
        "bottle": np.array([128, 0, 128]),
        "bus": np.array([0, 128, 128]),
        "car": np.array([128, 128, 128]),
        "cat": np.array([64, 0, 0]),
        "chair": np.array([192, 0, 0]),
        "cow": np.array([64, 128, 0]),
        "diningtable": np.array([192, 128, 0]),
        "dog": np.array([64, 0, 128]),
        "horse": np.array([192, 0, 128]),
        "motorbike": np.array([64, 128, 128]),
        "person": np.array([192, 128, 128]),
        "pottedplant": np.array([0, 64, 0]),
        "sheep": np.array([128, 64, 0]),
        "sofa": np.array([0, 192, 0]),
        "train": np.array([128, 192, 0]),
        "tv": np.array([0, 64, 128])
    }
    
    def __init__(self, image_names_file, image_folder, mask_folder, batch_size,
                 shape, augment=False, augmentation=None, repeat_num=1):
        self.image_names_file = image_names_file
        self.images = open(self.image_names_file).readlines()
        self.images = [name[:-1] for name in self.images]
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.batch_size = batch_size
        self.image_shape = shape
        self.augment = augment
        if self.augment:
            self.augmentation = augmentation
        
        self.images = np.repeat(self.images, repeat_num)
        
    def __len__(self):
        return int(np.ceil(len(self.images) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        batch_images = self.images[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = [cv2.imread('{}.jpg'.format(join(self.image_folder, _im))) for _im in batch_images]
        batch_y = [cv2.imread('{}.png'.format(join(self.mask_folder, _im))) for _im in batch_images]

        batch_x = np.array([cv2.resize(_im, self.image_shape)[:, :, ::-1] for _im in batch_x])
        batch_y = np.array([cv2.resize(_im, self.image_shape, cv2.INTER_NEAREST)[:, :, ::-1] for _im in batch_y])
        
        if self.augment:
            augmented = [
                self.augmentation(image=_im, mask=_m) for _im, _m in zip(batch_x, batch_y)
            ]
            batch_x = np.array([_aug['image'] for _aug in augmented])
            batch_y = np.array([_aug['mask'] for _aug in augmented])
        
        batch_y = np.array([self.mask_to_categorical(_im) for _im in batch_y])
        
        return batch_x, batch_y
    
    def mask_to_categorical(self, mask):
        cat = np.zeros((mask.shape[0], mask.shape[1], len(self.classes)), np.float)
        for i, (cls, color) in enumerate(self.classes.items()):
            cat[:, :, i] = (mask == color).all(axis=2).astype(np.float)
        return cat
    
    def categorical_to_mask(self, categorical):
        mask = np.zeros((categorical.shape[0], categorical.shape[1], 3), np.int)
        for i, (cls, color) in enumerate(self.classes.items()):
            mask[categorical[:, :, i] == 1.] = color
        return mask
    
    def on_epoch_end(self):
        np.random.shuffle(self.images)

        # Fix memory leak (Keras bug)
        gc.collect()

    @staticmethod
    def segmentation_plot(image, mask, name=None):
        """Return instance of matplotlib figure with 3 images:
        image, mask, mask on image.

        :image (array [N, M, 3]): source image
        :mask (array [N, M, 3]): source mask
        :return: instance of matplotlib.pyplot.figure
        """
        blend_image = image.copy()
        blend_mask = (mask != [0, 0, 0]).any(axis=2)
        blend_image[blend_mask] = (blend_image[blend_mask] * .5 + mask[blend_mask] * .5)

        fig, axs = plt.subplots(1, 3, figsize=(15, 15))
        for i in [0, 1, 2]:
            axs[i].grid(False)
            axs[i].axis('off')
        axs[0].imshow(image)
        axs[1].imshow(mask)
        axs[2].imshow(blend_image)
        if name is not None:
            axs[1].set_title(name)
        return fig