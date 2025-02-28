from collections import OrderedDict

import torch
import numpy as np
from fed.client import FedAvgClient
from inference import inference
W
class BaseServer:
    def __init__(self,
                 dataset: str,
                 train_loaders, test_loader,
                 clients_num: int,
                 global_rounds: int,
                 local_epochs: int,
                 model,
                 ):
        super().__init__()
        self.clients = []
        self.dataset = dataset
        self.train_loaders = train_loaders
        self.test_loader = test_loader
        self.global_rounds = global_rounds
        self.model = model
        self.local_epochs = local_epochs
        self.clients_num = clients_num
        self.local_weights = []
        self.local_losses = []
        self.len_dataset = []


class FedAvgServer(BaseServer):
    def __init__(self,
                 dataset: str,
                 train_loaders, test_loader,
                 clients_num: int,
                 global_rounds: int,
                 local_epochs: int,
                 learning_rate: float,
                 split_mode,
                 scheduler_milestones,
                 scheduler_rate,
                 device,
                 model
                 ):
        super().__init__(dataset, train_loaders, test_loader, clients_num, global_rounds, local_epochs, model)
        self.dataset = dataset
        self.train_loaders = train_loaders
        self.test_loader = test_loader
        self.global_rounds = global_rounds
        self.local_epochs = local_epochs
        self.split_mode = split_mode
        self.model = model
        self.global_parameter = None
        self.best = 0
        self.best_model = None
        self.device = device

        if self.dataset == 'ucf':
            label_map = dict(
                {'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson', 'Assault': 'assault',
                 'Burglary': 'burglary', 'Explosion': 'explosion', 'Fighting': 'fighting',
                 'RoadAccidents': 'roadAccidents', 'Robbery': 'robbery', 'Shooting': 'shooting',
                 'Shoplifting': 'shoplifting', 'Stealing': 'stealing', 'Vandalism': 'vandalism'})
        else:
            label_map = dict({'A': 'normal', 'B1': 'fighting', 'B2': 'shooting', 'B4': 'riot',
                              'B5': 'abuse', 'B6': 'car accident', 'G': 'explosion'})

        for i in range(clients_num):

            client = FedAvgClient(model, learning_rate, train_loaders[i], dataset,
                                  local_epochs, label_map, scheduler_milestones,
                                  scheduler_rate, device)

            self.clients.append(client)

    def aggregate_parameters(self):
        temp_dict = OrderedDict()
        total_num = sum(self.len_dataset)
        for key, value in self.local_weights[0].items():
            temp_dict[key] = torch.zeros_like(value)

        for i in range(len(self.local_weights)):
            for key, value in self.local_weights[i].items():
                temp_dict[key] += value * self.len_dataset[i] / total_num

        return temp_dict

    def set_global_parameter(self, para):
        state_dict = self.model.state_dict()
        for key, value in para.items():
            state_dict[key] = value.data.clone()
        self.model.load_state_dict(state_dict)

    def send_global_parameter(self, para):
        for client in self.clients:
            client.set_parameters(para)

    def evaluate(self, r):
        if self.dataset == 'ucf':
            gt = np.load("./data/gt_ucf.npy")
        else:
            gt = np.load("./data/gt_xd.npy")

        roc, ap = inference(self.dataset, self.model, self.test_loader, gt, self.device)
        print(f"round {r + 1} : roc: {roc} , ap: {ap}")
        res_dict = {
            'ucf': roc,
            'xd': ap
        }
        return res_dict

    def evaluate_local(self, r):
        if self.dataset == 'ucf':
            gt = np.load("./data/gt_ucf.npy")
        else:
            gt = np.load("./data/gt_xd.npy")

        res_dict = {
            'ucf': [],
            'xd': []
        }
        for index in range(len(self.local_weights)):
            self.set_global_parameter(self.local_weights[index])
            roc, ap = inference(self.dataset, self.model, self.test_loader, gt, self.device)
            res_dict['ucf'].append(roc)
            res_dict['xd'].append(ap)

        print(f"round {r + 1} : roc: {sum(res_dict['ucf']) / len(res_dict['ucf'])} ,"
              f" ap: {sum(res_dict['ap']) / len(res_dict['ap'])}")

    def train(self, dir_name):
        for g in range(self.global_rounds):
            print(f"-------- round: {g + 1} / {self.global_rounds} --------")
            self.local_weights.clear()
            self.local_losses.clear()
            self.len_dataset.clear()

            i = 0
            for client in self.clients:
                i += 1
                print(f"round {g + 1}, client: {i}")
                w, loss, l_data = client.train()

                self.local_weights.append(w)
                self.local_losses.append(loss)
                self.len_dataset.append(l_data)
                client.scheduler.step()

            self.global_parameter = self.aggregate_parameters()
            self.set_global_parameter(self.global_parameter)
            res = self.evaluate(g)

            self.send_global_parameter(self.global_parameter)
            if res[self.dataset] > self.best:
                self.best = res[self.dataset]
                self.best_model = self.global_parameter

                torch.save(self.model.state_dict(), dir_name + "/model.pth")

            print(f"best: {self.best}")

        self.set_global_parameter(self.best_model)
        torch.save(self.model.state_dict(), dir_name + "/model.pth")

