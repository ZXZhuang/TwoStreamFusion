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
    def __init__(self, train=True, transform=None, ratio=0.7, Spatial=True):
        self.size = (224, 224)
        self.transform = transform
        self.train = train
        self.ratio = ratio
        self.data = {}
        self.label_index = {}
        self.num = 3
        video_list = []
        #视频文件
        video_folder = './hmdb51'
        #存储数据文件
        data_folder = 'D:\\data'
        #数据分类路径
        path_list = [join(data_folder, 'train'), join(data_folder, 'validation'), join(data_folder, 'test')]
        #动作类型
        self.labels = sorted(os.listdir(join(video_folder)))#[a, b,c,d]

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
                if num > self.num:
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
            f_name = 'spatial'
        else:
            f_name = 'temporal'
        image_list = glob.glob(join(data_folder, mode, f_name, '**', '*.jpg'), recursive=True)
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

        _, frame = capture.read()
        prvs = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255

        while True:
            ret, image = capture.read()
            if not ret:
                break

            count += 1
            # if(int(capture.get(1)) % get_frame_rate == 0):
            fname = '{0}_{1:05d}_S.jpg'.format(video_name, count)
            cv2.imwrite(join(spatial_path, folder_name, fname), image)

            # if(int(capture.get(1)) % get_frame_rate == 0):
            next_ = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs, next_, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            fname = '{0}_{1:05d}_T.jpg'.format(video_name, count)
            cv2.imwrite(join(temporal_path, folder_name, fname), rgb)

            prvs = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        print('{} spatial images are extracted in {}'.format(count, join(spatial_path, folder_name, video_name)))
        print('{} temporal images are extracted in {}.'.format(count, join(temporal_path, folder_name, video_name)))

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

        img = Image.open(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, target


if __name__ == '__main__':
    HMDB51()

'''
    def video2frame(self, video, spatial_path, temporal_path, count=0):
        folder_name, video_name = video.split('\\')[-2], video.split('\\')[-1]

        capture = cv2.VideoCapture(video)
        ret, previous_frame = capture.read()

        if ret:
            # resize frame
            frame = cv2.resize(previous_frame, self.size)
            # convert to gray
            previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            previous_frame = np.float32(previous_frame) / 255.0

            # upload pre-processed frame to GPU
            gpu_previous = cv2.cuda_GpuMat(self.size, cv2.CV_32FC1)
            gpu_previous.upload(previous_frame)

            while True:
                ret, frame = capture.read()
                if not ret:
                    break

                count += 1
                fname = '{0}_{1:05d}_S.jpg'.format(video_name, count)
                cv2.imwrite(join(spatial_path, folder_name, fname), frame)

                frame = cv2.resize(frame, self.size)
                current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                current_frame = np.float32(current_frame) / 255.0

                gpu_current = cv2.cuda_GpuMat(self.size, cv2.CV_32FC1)
                gpu_current.upload(current_frame)

                gpu_flow = cv2.cuda_BroxOpticalFlow.create(0.197, 50.0, 0.8, 5, 150, 10)
                gpu_flow = cv2.cuda_BroxOpticalFlow.calc(gpu_flow, gpu_previous, gpu_current, None, )

                fname = '{0}_{1:05d}_T.jpg'.format(video_name, count)

                gpu_flow_x = cv2.cuda_GpuMat(gpu_flow.size(), cv2.CV_32FC1)
                gpu_flow_y = cv2.cuda_GpuMat(gpu_flow.size(), cv2.CV_32FC1)

                cv2.cuda.split(gpu_flow, [gpu_flow_x, gpu_flow_y])

                optical_flow_x = gpu_flow_x.download()
                optical_flow_y = gpu_flow_y.download()

                #norm
                dist_x = optical_flow_x.max() - optical_flow_x.min()
                dist_y = optical_flow_y.max() - optical_flow_x.min()
                img = np.zeros((224, 224, 3), dtype=np.int)
                for x in range(224):
                    for y in range(224):
                        img[x][y][0] = (optical_flow_x[x][y] - optical_flow_x.min()) * 255 / dist_x
                        img[x][y][1] = (optical_flow_y[x][y] - optical_flow_y.min()) * 255 / dist_y
                cv2.imwrite(join(temporal_path, folder_name, fname), img)

                gpu_previous.upload(current_frame)


        print('{} spatial images are extracted in {}'.format(count, join(spatial_path, folder_name, video_name)))
        print('{} temporal images are extracted in {}.'.format(count, join(temporal_path, folder_name, video_name)))
    '''