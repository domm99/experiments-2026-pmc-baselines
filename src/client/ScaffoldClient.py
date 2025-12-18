import copy
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.FedUtils import *

class ScaffoldClient:

    def __init__(self, mid, dataset_name, dataset, batch_size, epochs, sparsification_level):
        self.mid = mid
        self.lr = 0.001
        self.epochs = epochs
        self.weight_decay=1e-4
        self.training_set = dataset[0]
        self.batch_size = batch_size
        self.sparsification_level = sparsification_level
        self._model = initialize_model(dataset_name)
        self._model = prune_model(self._model.state_dict(), dataset_name, self.sparsification_level)
        # self.global_model = initialize_model(dataset_name)
        self.global_model = self._model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.server_control_state = initialize_control_state(dataset_name, self.device)
        self.server_control_state = prune_model(self.server_control_state, dataset_name, self.sparsification_level).to(self.device).state_dict()
        self._client_control_state = initialize_control_state(dataset_name, self.device)
        self._client_control_state = prune_model(self._client_control_state, dataset_name, self.sparsification_level).to(self.device).state_dict()

    def train(self):
        # labels = [self.training_set[idx][1] for idx in range(len(self.training_set))]
        # print(f'Client {self.mid} --> training set size {len(self.training_set)} classes {set(labels)}')
        train_loader = DataLoader(self.training_set, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        ccs_dict = self._client_control_state
        scs_dict = self.server_control_state
        loss_func = nn.CrossEntropyLoss()
        losses = []
        tau = 0
        self._model.to(self.device)
        for epoch in range(self.epochs):
            batch_losses = []
            for step, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                with (torch.enable_grad()):
                    self._model.train()
                    outputs = self._model(images)
                    loss = loss_func(outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    batch_losses.append(loss.item())
                    tau = tau + 1
                    model_dict = self._model.state_dict()#{k: v.to('cpu') for k, v in self._model.state_dict().items()}
                    for key in model_dict:
                        # Eq (3) in Scaffold paper but simplified since optimizer already computes the term (y_i - lr * g(y_i))
                        model_dict[key] = model_dict[key] - self.lr * (scs_dict[key] - ccs_dict[key])
                    self._model.load_state_dict(model_dict)
            mean_epoch_loss = sum(batch_losses)/len(batch_losses)
            losses.append(mean_epoch_loss)

        self.update_control_state(tau)

        return sum(losses)/len(losses)

    def update_control_state(self, tau):
        local_model_dict = self._model.state_dict() #{k: v.to('cpu') for k, v in self._model.state_dict().items()}
        global_model_dict = self.global_model.state_dict()#{k: v.to('cpu') for k, v in self.global_model.state_dict().items()}
        ccs_dict = self._client_control_state
        scs_dict = self.server_control_state
        for key in ccs_dict:
            # Option II of Eq (4) in Scaffold paper
            ccs_dict[key] = ccs_dict[key] - scs_dict[key] + ((1 / (tau * self.lr)) * (global_model_dict[key] - local_model_dict[key]))

    def notify_updates(self, global_model, server_control_state):
        self._model.load_state_dict(copy.deepcopy(global_model.state_dict()))
        self.global_model.load_state_dict(copy.deepcopy(global_model.state_dict()))
        self._model.to(self.device)
        self.global_model.to(self.device)
        self.server_control_state = server_control_state

    @property
    def model(self):
        return self._model

    @property
    def client_control_state(self):
        return self._client_control_state
