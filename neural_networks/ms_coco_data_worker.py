"""Script for working with data
Assume DATA_PATH - path to ms coco dataset, can be symlink
then folder structure should be:
DATA_PATH/
    train2017/
        1.jpg
        2.jpg
        ...
    val2017/
        3.jpg
        4.jpg
        ...
    annotations/
        instances_train2017.json
        instances_val2017.json
"""

import os
import cv2
import numpy as np
from pycocotools.coco import COCO
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

from utils.utils import resize_pad

class DataWorker:
    """Using for loading ms coco dataset (only train and val parts),
    filter data for person category (because of solving task),
    create generator for batch learning
    """

    train_folder = 'train2017'
    test_folder = 'val2017'
    annotation_folder = 'annotations'

    # person
    category_id = 1
    
    def __init__(self, data_path, seed=7):
        """Read images description and annotations about it

        :data_path (str): path to dataset's folder
        :seed (int): random seed
        """
        self.seed = seed
        if os.path.islink(data_path):
            data_path = os.readlink(data_path)
        self.data_path = data_path
        self.train_folder = os.path.join(data_path, self.train_folder)
        self.test_folder = os.path.join(data_path, self.test_folder)
        self.annotation_folder = os.path.join(data_path, self.annotation_folder)

        self.coco_train = COCO(os.path.join(self.annotation_folder, 'instances_train2017.json'))
        self.coco_test = COCO(os.path.join(self.annotation_folder, 'instances_val2017.json'))

        # load information about images with class label
        self.train_images = self.coco_train.loadImgs(
            ids=self.coco_train.getImgIds(
                catIds=self.category_id
            )
        )
        self.test_images = self.coco_test.loadImgs(
            ids=self.coco_test.getImgIds(
                catIds=self.category_id
            )
        )

        # load annotations for loaded images
        self.train_annotations = {
            img_desc['id']: self.coco_train.loadAnns(
                self.coco_train.getAnnIds(
                    imgIds=img_desc['id'], iscrowd=False, catIds=self.category_id
                )
            ) for img_desc in self.train_images
        }
        self.test_annotations = {
            img_desc['id']: self.coco_test.loadAnns(
                self.coco_test.getAnnIds(
                    imgIds=img_desc['id'], iscrowd=False, catIds=self.category_id
                )
            ) for img_desc in self.test_images
        }
        
        self.train_images, self.val_images = train_test_split(self.train_images, test_size=0.2)

    @property
    def train_shape(self):
        """Return train shape

        :return (int): return train shape
        """
        return len(self.train_images)
    
    @property
    def val_shape(self):
        """Return test shape

        :return (int): return test shape
        """
        return len(self.val_images)

    @property
    def test_shape(self):
        """Return test shape

        :return (int): return test shape
        """
        return len(self.test_images)

    def load_image_mask(self, image_desc):
        """Load image and corresponding mask

        :image_desc (dict): description of image in COCO format

        :return (tuple([N, M, 3], [N, M])): tuple of image and mask
        """
        # reorder, because cv2 use BGR format
        if image_desc['id'] in self.train_annotations:
            image = cv2.imread(os.path.join(self.train_folder, image_desc['file_name']))
            masks = [
                self.coco_train.annToMask(i_img_ann)
                for i_img_ann in self.train_annotations[image_desc['id']]
            ]
        else:
            image = cv2.imread(os.path.join(self.test_folder, image_desc['file_name']))
            masks = [
                self.coco_test.annToMask(i_img_ann)
                for i_img_ann in self.test_annotations[image_desc['id']]
            ]
        image = image[:, :, [2, 1, 0]]
        total_mask = np.bitwise_or.reduce(masks)
        return (image, total_mask)

    def batch_loader(self, images_descriptions, batch_size, height, width):
        """Load batches of images and resize with padding to given shape

        :images_descriptions (dict): descriptions of images in coco format
        :batch_size (int): size of batch
        :height (int): height of proccessed images
        :width (int): width of proccesses images

        :return (generator): generator of batches with images and masks
        tuple of [batch_size, height, width, 3] and [batch_size, height, width]
        """
        for start_ind in range(0, len(images_descriptions), batch_size):
            images = np.empty([0, height, width, 3], dtype=np.uint8)
            masks = np.empty([0, height, width], dtype=np.float32)
            for image_desc in images_descriptions[start_ind:start_ind + batch_size]:
                image, mask = self.load_image_mask(image_desc)
                shaped_image = resize_pad(image, height, width)
                shaped_mask = resize_pad(mask, height, width)

                images = np.append(images, [shaped_image], axis=0)
                masks = np.append(masks, [shaped_mask], axis=0)
            yield (images, masks)

    def batch_augmentation(self, image_generator, augment_args):
        """Augmentate batch of images

        :image_generator (generator): generator with batches of images
        tuple of [batch_size, height, width, 3] and [batch_size, height, width]
        :augment_args (dict): params for augmentation

        :return (generator): generator with batches of augmented images
        tuple of [batch_size, height, width, 3] and [batch_size, height, width]
        """
        augment = ImageDataGenerator(**augment_args)
        for images, masks in image_generator:
            stacked = np.concatenate([images, masks[:, :, :, np.newaxis]], axis=-1)
            aug_batch = augment.flow(stacked, seed=self.seed, batch_size=stacked.shape[0],
                                     shuffle=False)
            for aug_stacked in aug_batch:
                aug_images = aug_stacked[:, :, :, :3].astype(np.uint8)
                aug_masks = aug_stacked[:, :, :, 3]#[:, :, :, np.newaxis]
                yield (aug_images, aug_masks)

    def batch_generator(self, images_descriptions, batch_size=100, height=512, width=512,
                        augment_args=None):
        """Pipeline, which take image description and proccessing information,
        Load it by batches, augmented and normalize

        :images_descriptions (dict): descriptions of images in coco format
        :batch_size (int): size of batch
        :height (int): height of proccessed images
        :width (int): width of proccesses images
        :augment_args (dict): params for augmentation

        :return (generator): generator with batches of images
        tuple of [batch_size, height, width, 3] and [batch_size, height, width]
        """
        if augment_args is None:
            augment_args = {
                'rotation_range': 15, 'width_shift_range': 0.1, 'height_shift_range': 0.1,
                'zoom_range': 0.25, 'horizontal_flip': True, 'brightness_range': [0.75, 1.25],
                'fill_mode': 'constant'
            }
        batch_gen = self.batch_augmentation(
            self.batch_loader(images_descriptions, batch_size, height, width),
            augment_args
        )
        for images, masks in batch_gen:
            yield (images, masks)
