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

        #if self.test:
            #video_name = video_name.decode("utf-8")
        video_path = self.root_dir + video_name + '/'
        
        #now this is a singular mp4 video
        video_name = sorted(os.listdir(video_path))[0]
        video_path = video_path+video_name
        vidcap = cv2.VideoCapture(video_path)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = torch.FloatTensor(frame_count, self.channels, self.xSize, self.ySize)

        sec = 0.0
        frame_rate = 1/fps 
        
        for f in range(frame_count):
            vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
            has_frames,frame = vidcap.read()
            frame = torch.from_numpy(frame)
            # HWC to CHW
            frame = frame.permute(2, 0, 1)
            frames[f, :, :, :] = transform(frame)
            sec = sec + frame_rate
            sec = round(sec, 10)
       
        labels = self.seq_labels[self.labels_dict['idx'][0]:self.labels_dict['idx'][1], idx] / w.tolist()
        labels[labels > 1] = 1  # threshold anything bigger than 1 to 1
        labels[labels < 0] = 0  # threshold anything smaller than 0 to 0
        labels = torch.from_numpy(labels)
        sample = {'clip': frames, 'label': labels.float(), 'name': video_path}
        return sample['clip'], sample['label']