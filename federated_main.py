import time

import pickle

from Trainer import *
from model_network import TLEncoder
import torch
import os
from tqdm import tqdm
import numpy as np
import copy
from federated_update import LocalUpdate, average_weights
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Federated learning')
    parser.add_argument('--global_round', type=int, default=10, help='global training epoch')
    parser.add_argument('--clients', type=int, default=50, help='epochs')
    parser.add_argument('--dataset', type=str, default="tdrive", help='dataset for training')
    args = parser.parse_args()

    start_time = time.time()

    print(torch.__version__)
    print(torch.cuda.device_count())
    print(torch.cuda.is_available())

    global_config = STsim_Trainer(dataset=args.dataset)
    global_model = TLEncoder(feature_size=global_config.feature_size,
                                    embedding_size=global_config.embedding_size,
                                    date2vec_size=global_config.date2vec_size,
                                    hidden_size=global_config.hidden_size,
                                    num_layers=global_config.num_layers,
                                    dropout_rate=global_config.dropout_rate,
                                    concat=global_config.concat,
                                    device=global_config.device)
    global_model.to(global_config.device)
    global_model.train()
    print(global_model)
    global_weights = global_model.state_dict()

    BATCH_SIZE = 20

    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0
    global_model_name = ''
    global_model_save = None
    for epoch in tqdm(range(args.global_round)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch} |\n')

        global_model.train()
        m = max(args.clients, 1)
        m = int(1 * m)
        idxs_users = np.random.choice(range(args.clients), m, replace=False)
        for idx in idxs_users:
            local_model = LocalUpdate(dataset="tdrive_taxi_{}".format(idx%25+1))
            # local_model = LocalUpdate(dataset="tdrive_taxi_{}".format(idx%25+1))
            print('users:', idx)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

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
        global_model_save = './model/{}_{}_2w_ST/{}_{}_epoch:{}.pkl'.format(global_config.dataset,
                                                                            global_config.distance_type,
                                                                            global_config.dataset,
                                                                            global_config.distance_type, args.global_round)
        torch.save(global_model.state_dict(), global_model_save)

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
    test_model = LocalUpdate(dataset='tdrive')
    test_acc_1 = test_model.test_inference(load_model=global_model_save)
    # test_model_2 = LocalUpdate(dataset='rome')
    # test_acc_2 = test_model_2.test_inference(load_model=global_model_save)
    print(f' \n Results after {args.global_round} global rounds of training:')
    print("|test on tdrive---- HR@10_{},HR@50_{},R10@50_{}".format(test_acc_1[0], test_acc_1[1], test_acc_1[2]))
    # print("|test on rome---- HR@10_{},HR@50_{},R10@50_{}".format(test_acc_2[0], test_acc_2[1], test_acc_2[2]))

    # Saving the objects train_loss and train_accuracy:
    # file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
    #     format(global_config.dataset, global_model, global_config.epochs, args.frac, args.iid,
    #            args.local_ep, args.local_bs)
    # with open(file_name, 'wb') as f:
    #     pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))

    # train and test

    #
    # STsim = STsim_Trainer()
    #
    # load_model_name = None
    # load_optimizer_name = None
    #
    # STsim.ST_train(load_model=load_model_name, load_optimizer=load_optimizer_name)

    # STsim.ST_eval(load_model=load_model_name)
