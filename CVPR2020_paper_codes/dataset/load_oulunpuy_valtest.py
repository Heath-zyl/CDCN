from __future__ import print_function, division
import os
import torch
import pandas as pd
#from skimage import io, transform
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pdb
import math
import os 


frames_total = 8    # each video 8 uniform samples
 
face_scale = 1.3  #default for test and val 


def pad_for_croped_map(image, scale):
    h, w = image.shape
    
    h_new, w_new = h * scale, w * scale
    h_pad, w_pad = int(h_new / 2.), int(w_new / 2.)

    new_image = cv2.copyMakeBorder(image, h_pad, h_pad, w_pad, w_pad, cv2.BORDER_CONSTANT, 0)

    return new_image


def crop_face_from_scene(image,face_name_full, scale):
    f=open(face_name_full,'r')
    lines=f.readlines()
    y1,x1,w,h=[float(ele) for ele in lines[:4]]
    f.close()
    y2=y1+w
    x2=x1+h

    y_mid=(y1+y2)/2.0
    x_mid=(x1+x2)/2.0
    h_img, w_img = image.shape[0], image.shape[1]
    #w_img,h_img=image.size
    w_scale=scale*w
    h_scale=scale*h
    y1=y_mid-w_scale/2.0
    x1=x_mid-h_scale/2.0
    y2=y_mid+w_scale/2.0
    x2=x_mid+h_scale/2.0
    y1=max(math.floor(y1),0)
    x1=max(math.floor(x1),0)
    y2=min(math.floor(y2),w_img)
    x2=min(math.floor(x2),h_img)

    #region=image[y1:y2,x1:x2]
    region=image[x1:x2,y1:y2]
    return region


class Normaliztion_valtest(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """
    def __call__(self, sample):
        image_x, val_map_x, spoofing_label = sample['image_x'],sample['val_map_x'] ,sample['spoofing_label']
        new_image_x = (image_x - 127.5)/128     # [-1,1]
        return {'image_x': new_image_x, 'val_map_x': val_map_x , 'spoofing_label': spoofing_label}


class ToTensor_valtest(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __call__(self, sample):
        image_x, val_map_x, spoofing_label = sample['image_x'],sample['val_map_x'] ,sample['spoofing_label']
        
        # swap color axis because    BGR2RGB
        # numpy image: (batch_size) x T x H x W x C
        # torch image: (batch_size) x T x C X H X W
        image_x = image_x[:,:,:,::-1].transpose((0, 3, 1, 2))
        image_x = np.array(image_x)
        
        val_map_x = np.array(val_map_x)
                        
        spoofing_label_np = np.array([0],dtype=np.long)
        spoofing_label_np[0] = spoofing_label
        
        return {'image_x': torch.from_numpy(image_x.astype(np.float)).float(), 'val_map_x': torch.from_numpy(val_map_x.astype(np.float)).float(),'spoofing_label': torch.from_numpy(spoofing_label_np.astype(np.long)).long()} 


class Fas_valtest(Dataset):
    def __init__(self, info_list, transform=None, mode='val'):
        self.transform = transform
        self.mode = mode

        with open(info_list, 'r') as f:
            self.infos = f.readlines()
        
        self.videos = set()
        for line in self.infos:
            face_path = line.split()[0].strip()
            video_path = '/'.join(face_path.split('/')[:-1])
            self.videos.add(video_path)
        self.videos = list(self.videos)

    def __getitem__(self, idx):
        video_path = self.videos[idx]

        if self.mode == 'val':
            map_path = video_path.replace('Dev_images', 'Dev_3D/Dev_images')
        elif self.mode == 'test':
            map_path = video_path.replace('Test_images', 'Test_3D/Test_images')

        image_x, val_map_x = self.get_images(video_path, map_path)

        fas_label = 1 if video_path[-1] == '1' else 0
            
        sample = {'image_x': image_x, 'val_map_x':val_map_x , 'spoofing_label': fas_label}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.videos)
    
    def get_images(self, video_path, map_path):
        files_total = len([name for name in os.listdir(video_path) if os.path.isfile(os.path.join(video_path, name))])//3
        interval = files_total // 10
        videoname = video_path.split('/')[-1].strip()

        image_x = np.zeros((frames_total, 256, 256, 3))
        val_map_x = np.ones((frames_total, 32, 32))

        # random choose 1 frame
        for ii in range(frames_total):
            image_id = ii*interval + 1 
            
            for temp in range(50):
                scene_id = "_%03d_scene" % image_id
                scene_name = videoname + scene_id + '.jpg'
                box_name = videoname + scene_id + '.dat'
                
                scene_path = os.path.join(video_path, scene_name)
                box_path = os.path.join(video_path, box_name)
                map_path_x = os.path.join(map_path, scene_name).replace('scene.jpg', 'face_dep.jpg')
                # map_path = scene_path.replace('scene.jpg', 'face_dep.jpg')

                if os.path.exists(box_path) and os.path.exists(map_path_x):    # some scene.dat are missing
                    map_x = cv2.imread(map_path_x, cv2.IMREAD_UNCHANGED)
                    if map_x is not None:
                        break
                    else:
                        image_id += 1
                else:
                    image_id += 1


            # RGB
            image_x_temp = cv2.imread(scene_path)

            if not os.path.exists(map_path_x):
                print('==> NOT EXIST: ', map_path_x)

            # gray-map
            val_map_x_temp = cv2.imread(map_path_x, 0)

            image_x[ii,:,:,:] = cv2.resize(crop_face_from_scene(image_x_temp, box_path, face_scale), (256, 256))
            # transform to binary mask --> threshold = 0 
            # temp = cv2.resize(crop_face_from_scene(val_map_x_temp, box_path, face_scale), (32, 32))
            temp = cv2.resize(pad_for_croped_map(val_map_x_temp, face_scale), (32, 32))
            np.where(temp < 1, temp, 1)
            
            # if val_map_x_temp is not None:
            #     temp = np.ones((32, 32))
            # else:
            #     temp = np.zeros((32, 32))
            val_map_x[ii,:,:] = temp

        # print(image_x.shape, val_map_x.shape)
        return image_x, val_map_x
