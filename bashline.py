import parser
import time
import argparse
import pickle
from K_medians import cluster
from Trainer import *
from model_network import STTrajSimEncoder
import torch
import os
from tqdm import tqdm
import numpy as np
import copy
from federated_update import LocalUpdate, average_weights
from baseline_network import FL_FC,FL_RNN,FL_GCN


if __name__ == '__main__':
    start_time = time.time()

    print(torch.__version__)
    print(torch.cuda.device_count())
    print(torch.cuda.is_available())

    global_config = Trainer_config(dataset="tdrive")
    global_model = FL_RNN(date2vec_size=global_config.date2vec_size,
                           embedding_size=global_config.embedding_size,
                           hidden_size=global_config.hidden_size,
                           num_layers=global_config.num_layers,
                           dropout_rate=global_config.dropout_rate,
                           device=global_config.device)
    # global_model = FL_GCN(feature_size=global_config.feature_size,
    #                       hidden_size=global_config.hidden_size,
    #                       num_layers=global_config.num_layers,
    #                       dropout_rate=global_config.dropout_rate,
    #                         date2vec_size=global_config.date2vec_size,
    #                        embedding_size =global_config.embedding_size,
    #                        device=global_config.device)

    global_model.to(global_config.device)
    global_model.train()
    print(global_model)
    global_weights = global_model.state_dict()

    global_round = 1
    NUM_CLIENTS = 50
    BATCH_SIZE = 20
    # cluster_client = args.cluster_client
    cluster_client = 10

    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0
    global_model_name = ''
    global_model_save = None
    cluster = cluster(cluster_client, data="tdrive", client_num=NUM_CLIENTS)
    for epoch in tqdm(range(global_round)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch} |\n')
        global_model.train()
        # m = max(NUM_CLIENTS, 1)
        m = max(cluster_client, 1)

        idxs_users = np.random.choice(range(cluster_client), m, replace=False)
        for idx in idxs_users:
            for i in range(len(cluster)):
                if cluster[i] == idx:
                    local_model = LocalUpdate(dataset="tdrive_taxi_{}".format(i%25+1))
                    # local_model = LocalUpdate(dataset="tdrive_taxi_{}".format(i%25+1))
                    print("local client:",i)
                    print('cluster client:', idx)
                    w, loss = local_model.update_weights(
                        model=copy.deepcopy(global_model), global_round=epoch)
                    local_weights.append(copy.deepcopy(w))
                    local_losses.append(copy.deepcopy(loss))

        print("finish global epoch:{},time:{}".format(epoch, time.time()-start_time))
        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        acc = []
        list_acc = []
        global_model.eval()
        global_config.dataset = "tdrive"
        global_model_save = './model/{}_{}_2w_ST/test_{}_{}_client:{}.pkl'.format(global_config.dataset,
                                                                            global_config.distance_type,
                                                                            global_config.dataset,
                                                                            "RNN", NUM_CLIENTS)
        torch.save(global_model.state_dict(), global_model_save)

        test_model = LocalUpdate(dataset="tdrive_TP")
        test_acc_1 = test_model.RNN_inference(load_model=global_model_save)
        test_model_2 = LocalUpdate(dataset="tdrive_DITA")
        test_acc_2 = test_model_2.RNN_inference(load_model=global_model_save)
        test_model_3 = LocalUpdate(dataset="tdrive_LCRS")
        test_acc_3 = test_model_3.RNN_inference(load_model=global_model_save)
        test_model_4 = LocalUpdate(dataset="tdrive_NetERP")
        test_acc_4 = test_model_4.RNN_inference(load_model=global_model_save)
        print(f' \n Results after {global_round} global rounds of training:')
        print("|test on tdrive distance tpye:TP---- HR@10_{},HR@50_{},R10@50_{}".format(test_acc_1[0], test_acc_1[1],
                                                                                      test_acc_1[2]))
        print("|test on tdrive distance tpye:DITA---- HR@10_{},HR@50_{},R10@50_{}".format(test_acc_2[0], test_acc_2[1],
                                                                                        test_acc_2[2]))
        print("|test on tdrive distance tpye:LCRS---- HR@10_{},HR@50_{},R10@50_{}".format(test_acc_3[0], test_acc_3[1],
                                                                                        test_acc_3[2]))
        print("|test on tdrive distance tpye:NetERP---- HR@10_{},HR@50_{},R10@50_{}".format(test_acc_4[0], test_acc_4[1],
                                                                                          test_acc_4[2]))

        # for c in range(NUM_CLIENTS):
        #     local_model = LocalUpdate(datasets='tdrive')
        #     acc = local_model.inference(load_model=global_model_save)
        #     list_acc.append(acc)

        # train_accuracy.append(sum(list_acc[0]) / len(list_acc[0]))
        # print global training loss after every 'i' rounds
        if (epoch + 1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            # print('HR10,HR50,HR1050', train_accuracy)

    # Test inference after completion of training
    test_model = LocalUpdate(dataset="tdrive_TP")
    test_acc_1 = test_model.RNN_inference(load_model=global_model_save)
    test_model_2 = LocalUpdate(dataset="tdrive_DITA")
    test_acc_2 = test_model_2.RNN_inference(load_model=global_model_save)
    test_model_3 = LocalUpdate(dataset="tdrive_LCRS")
    test_acc_3 = test_model_3.RNN_inference(load_model=global_model_save)
    test_model_4 = LocalUpdate(dataset="tdrive_NetERP")
    test_acc_4 = test_model_4.RNN_inference(load_model=global_model_save)
    print(f' \n Results after {global_round} global rounds of training:')
    print("|test on tdrive distance tpye:TP---- HR@10_{},HR@50_{},R10@50_{}".format(test_acc_1[0], test_acc_1[1], test_acc_1[2]))
    print("|test on tdrive distance tpye:DITA---- HR@10_{},HR@50_{},R10@50_{}".format(test_acc_2[0], test_acc_2[1], test_acc_2[2]))
    print("|test on tdrive distance tpye:LCRS---- HR@10_{},HR@50_{},R10@50_{}".format(test_acc_3[0], test_acc_3[1], test_acc_3[2]))
    print("|test on tdrive distance tpye:NetERP---- HR@10_{},HR@50_{},R10@50_{}".format(test_acc_4[0], test_acc_4[1], test_acc_4[2]))