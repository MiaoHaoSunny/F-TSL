import torch
from torch import nn
import yaml
import data_utils
from lossfun import LossFun
import test_method
import copy
from model_network import STTrajSimEncoder
from data_utils import get_taxi_config,get_rome_taxi_config


class LocalUpdate(object):
    def __init__(self,dataset):
        if dataset == "tdrive":
            config = yaml.safe_load(open('config.yaml'))
        elif dataset =="rome":
            config = yaml.safe_load(open('config_rome.yaml'))
        elif dataset[0:6] == "tdrive":
            config = get_taxi_config(dataset.split('_')[2])
        elif dataset[0:4] == "rome":
            config = get_rome_taxi_config(dataset.split('_')[2])
        else:
            return

        self.feature_size = config["feature_size"]
        self.embedding_size = config["embedding_size"]
        self.date2vec_size = config["date2vec_size"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.dropout_rate = config["dropout_rate"]
        self.concat = config["concat"]
        self.device = "cuda:" + str(config["cuda"])
        self.learning_rate = config["learning_rate"]
        self.epochs = config["epochs"]

        self.train_batch = config["train_batch"]
        self.test_batch = config["test_batch"]
        self.traj_file = str(config["traj_file"])
        self.time_file = str(config["time_file"])

        self.dataset = str(config["dataset"])
        self.distance_type = str(config["distance_type"])
        self.early_stop = config["early_stop"]
        self.config = config

    def update_weights(self, model, global_round):
        # Set mode to train model
        epoch_loss = []

        dataload = data_utils.DataLoader(dataset=self.dataset)
        dataload.get_triplets()
        data_utils.triplet_groud_truth(dataset=self.dataset)

        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=self.learning_rate,
                                     weight_decay=0.0001)
        lossfunction = LossFun(self.train_batch, self.distance_type,dataset=self.dataset)

        model.to(self.device)
        lossfunction.to(self.device)

        road_network = data_utils.load_network(self.dataset).to(self.device)

        bt_num = int(dataload.return_triplets_num() / self.train_batch)

        batch_l = data_utils.batch_list(batch_size=self.train_batch, config = self.config)

        best_epoch = 0
        best_hr10 = 0
        lastepoch = '0'
        print('global round:',global_round)
        for epoch in range(int(lastepoch), self.epochs):
            batch_loss = []
            model.train()
            for bt in range(bt_num):
                a_node_batch, a_time_batch, p_node_batch, p_time_batch, n_node_batch, n_time_batch, batch_index = batch_l.getbatch_one()

                a_embedding = model(road_network, a_node_batch, a_time_batch)
                p_embedding = model(road_network, p_node_batch, p_time_batch)
                n_embedding = model(road_network, n_node_batch, n_time_batch)

                loss = lossfunction(a_embedding, p_embedding, n_embedding, batch_index)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_loss.append(loss.item())
            print('local_epochs:',epoch,loss.item())
            # if epoch % 2 == 0:
            #     model.eval()
            #     with torch.no_grad():
            #         vali_node_list, vali_time_list, vali_d2vec_list = dataload.load(load_part='vali')
            #         embedding_vali = test_method.compute_embedding(road_network=road_network, net=model,
            #                                                        test_traj=list(vali_node_list),
            #                                                        test_time=list(vali_d2vec_list),
            #                                                        test_batch=self.test_batch)
            #         acc = test_method.test_model(embedding_vali, isvali=True)
            #         print('local epoch:', epoch, acc[0], acc[1], acc[2], loss.item())

        epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(),sum(epoch_loss)/len(epoch_loss)


    def inference(self,load_model):
        model = STTrajSimEncoder(feature_size=self.feature_size,
                               embedding_size=self.embedding_size,
                               date2vec_size=self.date2vec_size,
                               hidden_size=self.hidden_size,
                               num_layers=self.num_layers,
                               dropout_rate=self.dropout_rate,
                               concat=self.concat,
                               device=self.device)
        if load_model != None:
            model.load_state_dict(torch.load(load_model))
            model.to(self.device)

            dataload = data_utils.DataLoader(dataset=self.dataset)
            road_network = data_utils.load_network(self.dataset).to(self.device)

            with torch.no_grad():
                vali_node_list, vali_time_list, vali_d2vec_list = dataload.load(load_part='vali')
                embedding_vali = test_method.compute_embedding(road_network=road_network, net=model,
                                                               test_traj=list(vali_node_list),
                                                               test_time=list(vali_d2vec_list),
                                                               test_batch=self.test_batch)
                acc = test_method.test_model(embedding_vali, isvali=True,datasets=self.dataset)
                print(acc)
        return acc

    def test_inference(self,load_model):
        model = STTrajSimEncoder(feature_size=self.feature_size,
                               embedding_size=self.embedding_size,
                               date2vec_size=self.date2vec_size,
                               hidden_size=self.hidden_size,
                               num_layers=self.num_layers,
                               dropout_rate=self.dropout_rate,
                               concat=self.concat,
                               device=self.device)
        if load_model != None:
            model.load_state_dict(torch.load(load_model))
            model.to(self.device)

            dataload = data_utils.DataLoader(dataset=self.dataset)
            road_network = data_utils.load_network(self.dataset).to(self.device)

            with torch.no_grad():
                vali_node_list, vali_time_list, vali_d2vec_list = dataload.load(load_part='test')
                embedding_vali = test_method.compute_embedding(road_network=road_network, net=model,
                                                               test_traj=list(vali_node_list),
                                                               test_time=list(vali_d2vec_list),
                                                               test_batch=self.test_batch)
                acc = test_method.test_model(embedding_vali, isvali=False,datasets=self.dataset)
                print(acc)
        return acc[0],acc[1],acc[2]




def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg