from __future__ import print_function
import torch
from torchvision import transforms
import numpy as np
import os
from torch.utils.data import Dataset
import cv2

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


class PainDataset(Dataset):
    def __init__(self, root_dir, channels, timeDepth, xSize, ySize, turn,files,labels_dict,test,mean=None):

        # (200,) the dir path for each sequence
        self.seq_labels = np.load(files['seq_labels'])
        self.video_paths = np.load(files['video_paths'])

        self.root_dir = root_dir
        self.timeDepth = timeDepth
        self.channels = channels
        self.xSize = xSize
        self.ySize = ySize
        self.mean = mean
        self.labels_dict = labels_dict
        self.test = test

    def __len__(self):
        return len(self.video_paths)  # 200 videos

    def __getitem__(self, idx):
        w = self.labels_dict['w']
        video_name = self.video_paths[idx]

        if self.test:
            video_name = video_name.decode("utf-8")
        video_path = self.root_dir + video_name + '/'
        frame_names = sorted(os.listdir(video_path))
        frames = torch.FloatTensor(len(frame_names), self.channels, self.xSize, self.ySize)

        for f in range(len(frame_names)):
            #print(video_path + frame_names[f])
            frame = cv2.imread(video_path + frame_names[f])
            frame = torch.from_numpy(frame)
            # HWC to CHW
            frame = frame.permute(2, 0, 1)
            frames[f, :, :, :] = transform(frame)

        labels = self.seq_labels[self.labels_dict['idx'][0]:self.labels_dict['idx'][1], idx] / w.tolist()
        labels[labels > 1] = 1  # threshold anything bigger than 1 to 1
        labels[labels < 0] = 0  # threshold anything smaller than 0 to 0
        labels = torch.from_numpy(labels)
        sample = {'clip': frames, 'label': labels.float(), 'name': video_path}
        return sample['clip'], sample['label']