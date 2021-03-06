import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import os
from os.path import join
import numpy as np
import glob
import cv2
import random
from PIL import Image


class HMDB51(Dataset):
    def __init__(self, train=True, transform=None, ratio=0.7, Spatial=True,):
        self.sequence_num = 10
        self.transform = transform
        self.train = train
        self.ratio = ratio
        self.data = {}
        self.label_index = {}
        self.class_num = 4
        self.Net_type = ''
        self.resize_shape = (256, 256)
        self.crop_shape = (224, 224)
        video_list = []
        #origin file dir
        video_folder = './hmdb51'
        #process data dir
        data_folder = './data'
        #data dir path
        path_list = [join(data_folder, 'train'), join(data_folder, 'validation'), join(data_folder, 'test')]
        #class
        self.labels = sorted(os.listdir(join(video_folder)))

        # label indexing, {'brush_hair': array(0)}, ...}
        self.label_index = {label: np.array(i) for i, label in enumerate(self.labels)}
        #video2image
        if not os.path.exists(join(data_folder, 'train')):
            for label in self.labels:
                video_list.append([avi for avi in glob.iglob(join(video_folder, label, '*.avi'), recursive=True)])

                for path in path_list:
                    os.makedirs(join(path, 'spatial', label), exist_ok=True)
                    os.makedirs(join(path, 'temporal', label), exist_ok=True)

            # len(video_list) = 51, len(videos) = how many videos in each label
            num = 0
            for videos in video_list:
                if num >= self.class_num:
                    break
                train_num = round(len(videos) * (self.ratio ** 2))
                test_num = round(len(videos) * (1 - self.ratio))
                for i, video in enumerate(videos):
                    if i < train_num:
                        self.video2frame(video, join(path_list[0], 'spatial'), join(path_list[0], 'temporal'))
                    elif train_num <= i < (len(videos) - test_num):
                        self.video2frame(video, join(path_list[1], 'spatial'), join(path_list[1], 'temporal'))
                    else:
                        self.video2frame(video, join(path_list[2], 'spatial'), join(path_list[2], 'temporal'))
                num+=1
        # {image: label}
        if train:
            mode = 'train'
        else:
            mode = 'test'
        if Spatial:
            self.Net_type = 'spatial'
            file_type = '*.jpg'
        else:
            self.Net_type = 'temporal'
            file_type = '*.npy'
        image_list = glob.glob(join(data_folder, mode, self.Net_type, '**', file_type), recursive=True)
        for image in image_list:
            self.data[image] = self.label_index[image.split('\\')[-2]]

        split_idx = int(len(image_list) * ratio)
        random.shuffle(image_list)
        self.train_image, self.test_image = image_list[:split_idx], image_list[split_idx:]

        self.train_label = [self.data[image] for image in self.train_image]
        self.test_label = [self.data[image] for image in self.test_image]

    def video2frame(self, video, spatial_path, temporal_path, count=0):
        folder_name, video_name = video.split('\\')[-2], video.split('\\')[-1]
        capture = cv2.VideoCapture(video)

        ret, prev_frame = capture.read()
        prev_frame = cv2.resize(prev_frame, self.resize_shape)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        #shape h * w
        frame_shape = prev_frame.shape[0:-1]
        # should be h * w * 2 * (L - 1) -> h * w * 18
        output_shape = frame_shape + (2 * (self.sequence_num - 1),)
        flows = np.ndarray(shape=output_shape)

        while(ret):
            for i in range(self.sequence_num - 1):
                ret, next_frame = capture.read()
                # video end
                if not ret:
                    break
                next_frame = cv2.resize(next_frame, self.resize_shape)
                count += 1
                fname = '{0}_{1:05d}_S.jpg'.format(video_name, count)
                cv2.imwrite(join(spatial_path, folder_name, fname), next_frame)
                next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

                flow = cv2.calcOpticalFlowFarneback(prev_gray, prev_gray, flow=None, pyr_scale=0.5, levels=3, winsize=15,
                                                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

                flows[:, :, 2 * i:2 * i + 2] = flow
                prev_gray = next_gray
                if i == self.sequence_num - 2:
                    fname = '{0}_{1:05d}_T.npy'.format(video_name, count)
                    np.save(join(temporal_path, folder_name, fname), flows.astype(np.float32))

        print('{} spatial images are extracted in {}'.format(count, join(spatial_path, folder_name, video_name)))
        print('{} temporal images are extracted in {}.'.format(count, join(temporal_path, folder_name, video_name)))

    def process_clip(self, arr):
        flip = random.randrange(2) == 1
        x = random.randrange(arr.shape[0] - self.crop_shape[0])
        y = random.randrange(arr.shape[1] - self.crop_shape[1])

        #crop
        arr = arr[x:x+self.crop_shape[1], y:y+self.crop_shape[1], :]
        #flip
        if flip:
            arr = arr[:, ::-1, :]
        return arr

    def __len__(self):
        if self.train:
            return len(self.train_image)
        else:
            return len(self.test_image)

    def __getitem__(self, idx):
        if self.train:
            img, target = self.train_image[idx], self.train_label[idx]
        else:
            img, target = self.test_image[idx], self.test_label[idx]

        if self.Net_type == 'spatial':
            img = Image.open(img)

        elif self.Net_type == 'temporal':
            img = np.load(img)
            img = self.process_clip(img.astype(np.float32))



        if self.transform is not None:
            img = self.transform(img.copy())

        return img, target


if __name__ == '__main__':
    HMDB51()