import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math
import yaml

def compute_embedding(net, road_network, test_traj, test_time, test_batch):

    if len(test_traj) <= test_batch:
        embedding = net(road_network, test_traj, test_time)
        return embedding
    else:
        i = 0
        all_embedding = []
        while i < len(test_traj):
            embedding = net(road_network, test_traj[i:i+test_batch], test_time[i:i+test_batch])
            all_embedding.append(embedding)
            i += test_batch

        all_embedding = torch.cat(all_embedding,0)
        return all_embedding

def test_model(embedding_set,datasets,isvali=False):
    if datasets == "tdrive_TP":
        config = yaml.safe_load(open('config_TP.yaml'))
    elif datasets == "tdrive_DITA":
        config = yaml.safe_load(open('config_DITA.yaml'))
    elif datasets == "tdrive_LCRS":
        config = yaml.safe_load(open('config_LCRS.yaml'))
    elif datasets == "tdrive_NetERP":
        config = yaml.safe_load(open('config_NetERP.yaml'))
    elif datasets == "rome_TP":
        config = yaml.safe_load(open('config_rome_TP.yaml'))
    elif datasets == "rome_DITA":
        config = yaml.safe_load(open('config_rome_DITA.yaml'))
    elif datasets == "rome_LCRS":
        config = yaml.safe_load(open('config_rome_LCRS.yaml'))
    elif datasets == "rome_NetERP":
        config = yaml.safe_load(open('config_rome_NetERP.yaml'))

    if isvali==True:
        input_dis_matrix = np.load(str(config["path_vali_truth"]))
    else:
        input_dis_matrix = np.load(str(config["path_test_truth"]))

    embedding_set = embedding_set.data.cpu().numpy()
    print(embedding_set.shape)

    embedding_dis_matrix = []
    for t in embedding_set:
        emb = np.repeat([t], repeats=len(embedding_set), axis=0)
        matrix = np.linalg.norm(emb-embedding_set, ord=2, axis=1)
        embedding_dis_matrix.append(matrix.tolist())

    l_recall_10 = 0
    l_recall_50 = 0
    l_recall_10_50 = 0

    f_num = 0

    for i in range(len(input_dis_matrix)):
        input_r = np.array(input_dis_matrix[i])
        one_index = []
        for idx, value in enumerate(input_r):
            if value != -1:
                one_index.append(idx)
        input_r = input_r[one_index]
        input_r = input_r[:5000]

        input_r50 = np.argsort(input_r)[1:51]
        input_r10 = input_r50[:10]

        embed_r = np.array(embedding_dis_matrix[i])
        embed_r = embed_r[one_index]
        embed_r = embed_r[:5000]

        embed_r50 = np.argsort(embed_r)[1:51]
        embed_r10 = embed_r50[:10]

        if len(one_index)>=51:
            f_num += 1
            l_recall_10 += len(list(set(input_r10).intersection(set(embed_r10))))
            l_recall_50 += len(list(set(input_r50).intersection(set(embed_r50))))
            l_recall_10_50 += len(list(set(input_r50).intersection(set(embed_r10))))

    recall_10 = float(l_recall_10) / (10 * f_num)
    recall_50 = float(l_recall_50) / (50 * f_num)
    recall_10_50 = float(l_recall_10_50) / (10 * f_num)

    return recall_10, recall_50, recall_10_50


def prediction(x_pred, y_pred, prediction):
    criterion = torch.nn.CrossEntropyLoss()
    loss_sum = []
    for i in range(len(prediction)):
        loss = criterion(prediction[i],y_pred[i])
        loss_sum.append(loss)

    return  loss_sum