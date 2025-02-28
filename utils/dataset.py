import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader

# sys.path.append(os.path.dirname(__file__))
from utils.tools import process_feat, process_split


class UCFDataset(data.Dataset):
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool, normal: bool = False):
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.normal = normal
        if normal == True and test_mode == False:
            self.df = self.df.loc[self.df['label'] == 'Normal']
            self.df = self.df.reset_index()

        elif test_mode == False:
            self.df = self.df.loc[self.df['label'] != 'Normal']
            self.df = self.df.reset_index()

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        clip_feature = np.load(self.df.loc[index]['path'])
        if self.test_mode == False:
            clip_feature, clip_length = process_feat(clip_feature, self.clip_dim)
        else:
            clip_feature, clip_length = process_split(clip_feature, self.clip_dim)

        clip_feature = torch.tensor(clip_feature)
        clip_label = self.df.loc[index]['label']
        return clip_feature, clip_label, clip_length


class XDDataset(data.Dataset):
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool):
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        clip_feature = np.load(self.df.loc[index]['path'])
        if not self.test_mode:
            clip_feature, clip_length = process_feat(clip_feature, self.clip_dim)
        else:
            clip_feature, clip_length = process_split(clip_feature, self.clip_dim)

        clip_feature = torch.tensor(clip_feature)
        clip_label = self.df.loc[index]['label']
        return clip_feature, clip_label, clip_length


def make_ucf_dataloader(split_mode: str, clients_num: int, batch_size: int, visual_length: int):
    split_list = []
    if split_mode == "event":
        clients_num = 13
        split_list = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 'Fighting', 'RoadAccidents',
                      'Robbery', 'Shooting', 'Shoplifting', 'Stealing', 'Vandalism']
    elif split_mode == "scene":
        split_list = ['Street', 'Store', 'Office', 'Parking lot', 'Room',
                      'Restaurant', 'Bank', 'Factory', 'Gas station']
        clients_num = 9

    elif split_mode == "random":
        split_list = [f'{clients_num}_{i}' for i in range(clients_num)]

    train_loaders = []
    for i in range(clients_num):
        train_list = f"./data/list/ucf_{split_mode}/ucf_{split_list[i]}.csv"
        normal_dataset = UCFDataset(visual_length, train_list, False, True)
        normal_dataloader = DataLoader(normal_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        anomaly_dataset = UCFDataset(visual_length, train_list, False, False)
        anomaly_loader = DataLoader(anomaly_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        train_loader = (normal_dataloader, anomaly_loader)
        train_loaders.append(train_loader)

    test_list = './data/list/ucf_test.csv'
    test_dataset = UCFDataset(visual_length, test_list, True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loaders, test_loader


def make_xd_dataloader(split_mode: str, clients_num: int, batch_size: int, visual_length: int):
    split_list = []
    if split_mode == "event":
        clients_num = 6
        split_list = ['Fighting', 'Riot', 'Abuse', 'Shooting', 'Explosion', 'Car accident']

    elif split_mode == "scene":
        split_list = ['Street', 'Park', 'Sports', 'Highway', 'Living room', 'Wild', 'Factory',
                      'Mall and Store', 'Office', 'Theater', 'Restaurant', 'Vehicle', 'Classroom']
        clients_num = 13

    elif split_mode == "random":
        split_list = [f'{clients_num}_{i}' for i in range(clients_num)]

    train_loaders = []
    for i in range(clients_num):
        train_list = f"./data/list/xd_{split_mode}/xd_{split_list[i]}.csv"
        train_dataset = XDDataset(visual_length, train_list, False)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        train_loaders.append(train_loader)

    test_list = './data/list/xd_test.csv'
    test_dataset = XDDataset(visual_length, test_list, True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return train_loaders, test_loader
