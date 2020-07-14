import csv
import os
import random

import numpy as np
from PIL import Image


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def gray_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('P')


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolder(object):
    def __init__(self, data_root="", mode="train", transform=None, loader=default_loader, gray_loader=gray_loader,
                 episode_num=1000, way_num=5, shot_num=5, query_num=5):
        super(ImageFolder, self).__init__()
        assert mode in ['train', 'val', 'test']

        self.mode = mode
        self.data_root = data_root

        self.episode_num = episode_num
        self.way_num = way_num
        self.shot_num = shot_num
        self.query_num = query_num

        self.transform = transform
        self.loader = loader
        self.gray_loader = gray_loader

        self.data_list = self._generate_episode_list()

    def _generate_episode_list(self):
        meta_csv = os.path.join(self.data_root, '{}.csv'.format(self.mode))

        data_list = []
        class_img_dict = {}
        with open(meta_csv) as f_csv:
            f_train = csv.reader(f_csv, delimiter=',')
            for row in f_train:
                if f_train.line_num == 1:
                    continue
                img_name, img_class = row

                class_img_dict.setdefault(img_class, []).append(img_name)

        class_list = class_img_dict.keys()
        for episode_idx in range(self.episode_num):
            # construct each episode
            episode = []
            temp_list = random.sample(class_list, self.way_num)

            for class_idx, class_name in enumerate(temp_list):
                image_list = class_img_dict[class_name]
                support_images = random.sample(image_list, self.shot_num)
                query_images = [val for val in image_list if val not in support_images]

                if self.query_num < len(query_images):
                    query_images = random.sample(query_images, self.query_num)

                # the dir of support_data set
                query_list = [os.path.join(self.data_root, 'images', path) for path in query_images]
                support_list = [os.path.join(self.data_root, 'images', path) for path in support_images]

                data_files = {
                    "query_list": query_list,
                    "support_list": support_list,
                    "target": class_idx
                }
                episode.append(data_files)
            data_list.append(episode)

        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        episode_files = self.data_list[index]

        query_images = []
        query_targets = []
        support_images = []
        support_targets = []

        for data_files in episode_files:
            query_list = data_files['query_list']

            for query in query_list:
                temp_image = self.loader(query)
                if self.transform is not None:
                    temp_image = self.transform(temp_image)
                query_images.append(temp_image)

            # load support_data images
            temp_support = []
            support_list = data_files['support_list']
            for support in support_list:
                temp_image = self.loader(support)
                if self.transform is not None:
                    temp_image = self.transform(temp_image)
                temp_support.append(temp_image)
            support_images.append(temp_support)

            # read the label
            target = data_files['target']
            query_targets.extend(np.tile(target, len(query_list)))
            support_targets.extend(np.tile(target, len(support_list)))

        return query_images, query_targets, support_images, support_targets
