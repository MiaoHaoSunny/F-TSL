from itertools import zip_longest

import numpy as np
import random

import pandas as pd

from data_utils import DataLoader


def divide_cluster(data_set, centorids):
    data_set_nums = np.shape(data_set)[0]
    centorids_nums = np.shape(centorids)[0]
    cluster = np.zeros(data_set_nums)
    for data_set_point in range(data_set_nums):
        distance = np.zeros(centorids_nums)
        for centorids_point in range(centorids_nums):
            distance[centorids_point] = np.linalg.norm(data_set[data_set_point]-centorids[centorids_point], ord=1)  # L1范数
        cluster[data_set_point] = np.argmin(distance)
    return cluster

def renew_centorids(data_set, cluster, k):
    centorids = np.zeros((k, np.shape(data_set)[1]))
    for centorid_id in range(k):
        bool_array = cluster == centorid_id
        centorids[centorid_id] = np.median(data_set[bool_array.flatten(), :], axis=0)
    return centorids


def kmedians(data_set, k, max_iterations):
    # 随机选取k个初始的中心点
    data_index = list(range(len(data_set)))
    random.shuffle(data_index)
    init_centorids_index = data_index[:k]
    centorids = data_set[init_centorids_index, :]  # k个初始中心点
    cluster = np.zeros(np.shape(data_set)[0])  # 用于标识每个样本点属于哪一个簇

    for _ in range(max_iterations):
        # 计算距离实现划分
        cluster = divide_cluster(data_set, centorids)
        # 更新中心点
        centorids = renew_centorids(data_set, cluster, k)

    return centorids, cluster


def cluster(k, data, client_num):
    dataset = []
    for i in range(0,client_num):
        Dataloader = DataLoader(dataset="{}_taxi_{}".format(data,i%25+1))
        train_dataset, _, _ = Dataloader.load(load_part="train")
        list_1 = []
        for j in range(len(train_dataset)):
            list_2 = train_dataset[j]
            list_1 = [x + y for x, y in zip_longest(list_1, list_2, fillvalue=0)]
        list_1 = [x/len(train_dataset) for x in list_1]
        dataset.append(list_1)
    dataset = pd.DataFrame(dataset)


    dataset = dataset.fillna(0, inplace=False)

    train_data = np.array(dataset)

    min_vals = train_data.min(0)
    max_vals = train_data.max(0)
    ranges = max_vals - min_vals
    normal_data = np.zeros(np.shape(train_data))
    nums = train_data.shape[0]
    normal_data = train_data - np.tile(min_vals, (nums, 1))
    normal_data = normal_data / np.tile(ranges, (nums, 1))

    max_iterations = 50
    centroids, cluster = kmedians(normal_data, k, max_iterations)
    return cluster