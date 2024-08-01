from torch.nn import Module, Parameter
import torch
import numpy as np
import math
import yaml
import pandas as pd
from data_utils import get_taxi_config,get_rome_taxi_config


class LossFun(Module):
    def __init__(self,train_batch,distance_type,dataset):
        super(LossFun, self).__init__()
        self.train_batch = train_batch
        self.distance_type = distance_type
        if dataset == "tdrive":
            config = yaml.safe_load(open('config.yaml'))
        elif dataset =="rome":
            config = yaml.safe_load(open('config_rome.yaml'))
        elif dataset[0:6] == "tdrive":
            config = get_taxi_config(dataset.split('_')[2])
        elif dataset[0:4] == "rome":
            config = get_rome_taxi_config(dataset.split('_')[2])

        self.triplets_dis = np.load(str(config["path_triplets_truth"]))

    def forward(self,embedding_a,embedding_p,embedding_n,batch_index):

        batch_triplet_dis = self.triplets_dis[batch_index]
        batch_loss = 0.0

        for i in range(self.train_batch):

            D_ap = math.exp(-batch_triplet_dis[i][0])
            D_an = math.exp(-batch_triplet_dis[i][1])

            v_ap = torch.exp(-torch.dist(embedding_a[i], embedding_p[i], p=2))
            v_an = torch.exp(-torch.dist(embedding_a[i], embedding_n[i], p=2))

            loss_entire_ap = D_ap * ((D_ap - v_ap) ** 2)
            loss_entire_an = D_an * ((D_an - v_an) ** 2)

            oneloss = loss_entire_ap + loss_entire_an
            batch_loss += oneloss

        mean_batch_loss = batch_loss / self.train_batch
        sum_batch_loss = batch_loss

        return mean_batch_loss

