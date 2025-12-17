import torch
import random
import numpy as np
import pandas as pd
import utils.FedUtils as utils
from collections import Counter
from client.IFCAClient import IFCAClient
from server.IFCAServer import IFCAServer
from torchvision import datasets, transforms
from client.FedAvgClient import FedAvgClient
from client.FedProxyClient import FedProxyClient
from server.FedAvgServer import FedAvgServer
from client.ScaffoldClient import ScaffoldClient
from server.ScaffoldServer import ScaffoldServer
from torch.utils.data import Subset, random_split
from ProFed.partitioner import Environment, Region, download_dataset, split_train_validation, partition_to_subregions

class Simulator:

    def __init__(self, algorithm, partitioning, areas, dataset_name, n_clients, batch_size, local_epochs, data_folder, seed):
        self.seed = seed
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.dataset_name = dataset_name
        self.algorithm = algorithm
        self.training_data, self.validation_data, self.test_data = self.initialize_data()
        self.partitioning = partitioning
        self.areas = areas
        self.n_clients = n_clients
        self.export_path = f'{data_folder}/seed-{seed}_algorithm-{self.algorithm}_dataset-{dataset_name}_partitioning-{self.partitioning}_areas-{self.areas}_clients-{self.n_clients}'
        self.simulation_data = pd.DataFrame(columns=['Round','TrainingLoss', 'ValidationLoss', 'ValidationAccuracy'])
        self.clients = self.initialize_clients()
        self.server = self.initialize_server()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def seed_everything(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def start(self, global_rounds):
        for r in range(global_rounds):
            print(f'Starting global round {r} -- using device {self.device}')
            self.notify_clients()
            training_loss = self.clients_update()
            self.notify_server()
            self.server_update()
            print('validation')
            validation_loss, validation_accuracy = self.test_global_model()
            self.export_data(r, training_loss, validation_loss, validation_accuracy)
        self.test_global_model(False)
        self.save_data()

    def initialize_clients(self):
        client_data_mapping = self.map_client_to_data()
        if self.algorithm == 'fedavg':
            return [FedAvgClient(index, self.dataset_name, client_data_mapping[index], self.batch_size, self.local_epochs) for index in range(self.n_clients)]
        elif self.algorithm == 'scaffold':
            return [ScaffoldClient(index, self.dataset_name, client_data_mapping[index], self.batch_size, self.local_epochs) for index in range(self.n_clients)]
        elif self.algorithm == 'fedproxy':
            return [FedProxyClient(index, self.dataset_name, client_data_mapping[index], self.batch_size, self.local_epochs) for index in range(self.n_clients)]
        elif self.algorithm == 'ifca':
            return [IFCAClient(index, self.dataset_name, client_data_mapping[index], self.batch_size, self.local_epochs) for index in range(self.n_clients)]
        else:
            raise Exception(f'Algorithm {self.algorithm} not supported! Please check :)')

    def initialize_server(self):
        if self.algorithm == 'fedavg' or self.algorithm == 'fedproxy':
            return FedAvgServer(self.dataset_name)
        elif self.algorithm == 'scaffold':
            return ScaffoldServer(self.dataset_name)
        elif self.algorithm == 'ifca':
            return IFCAServer(self.dataset_name, self.areas)
        else:
            raise Exception(f'Algorithm {self.algorithm} not supported! Please check :)')

    def notify_clients(self):
        for client in self.clients:
            if self.algorithm == 'fedavg' or self.algorithm == 'fedproxy' or self.algorithm == 'ifca':
                client.notify_updates(self.server.model)
            elif self.algorithm == 'scaffold':
                client.notify_updates(self.server.model, self.server.control_state)

    def clients_update(self):
        training_losses = []
        for client in self.clients:
            train_loss = client.train()
            training_losses.append(train_loss)
        average_training_loss = sum(training_losses) / len(training_losses)
        return average_training_loss

    def notify_server(self):
        client_data = {}
        for index, client in enumerate(self.clients):
            if self.algorithm == 'fedavg' or self.algorithm =='fedproxy' or self.algorithm == 'ifca':
                client_data[index] = client.model
            elif self.algorithm == 'scaffold':
                client_data[index] = { 'model': client.model, 'client_control_state': client.client_control_state }
        self.server.receive_client_update(client_data)

    def server_update(self):
        self.server.aggregate()

    def initialize_data(self):
        train, test_data = download_dataset(self.dataset_name)
        training_data, validation_data = split_train_validation(train, 0.8)
        if self.algorithm == 'ifca':
            test_data, _ = split_train_validation(test_data, 1.0)
        return training_data, validation_data, test_data

    def map_client_to_data(self) -> dict[int, Subset]:
        clients_split = np.array_split(list(range(self.n_clients)), self.areas)
        mapping_devices_area = { areaId: list(clients_split[areaId]) for areaId in range(self.areas) }

        environment = partition_to_subregions(self.training_data, self.validation_data, self.dataset_name, self.partitioning, self.areas, self.seed)

        mapping = {}
        for region_id, devices in mapping_devices_area.items():
            mapping_devices_data = environment.from_subregion_to_devices(region_id, len(devices))
            for device_index, data in mapping_devices_data.items():
                device_id = devices[device_index]
                mapping[device_id] = data # data is tuple(training_subset, validation_subset)

        if self.algorithm == 'ifca':
            environment_test = partition_to_subregions(self.test_data, self.test_data, 'CIFAR100', 'Hard', self.areas, self.seed)
            region_to_val_data = {}
            region_to_test_data = {}
            for region_id in range(self.areas):
                mapping_devices_data_val = environment.from_subregion_to_devices(region_id, 1)
                mapping_devices_data_test = environment_test.from_subregion_to_devices(region_id, 1)
                d_val = mapping_devices_data_val[0][0]
                d_test = mapping_devices_data_test[0][0]
                region_to_val_data[region_id] = Subset(d_val.dataset, d_val.indices)
                region_to_test_data[region_id] = Subset(d_test.dataset, d_test.indices)
                # print(f'---------------- region {region_id} ----------------')
                # labels = set([d_val.dataset[idx][1] for idx in d_val.indices])
                # print(f'Validation labels = {labels}')
                # labels = set([d_test.dataset[idx][1] for idx in d_test.indices])
                # print(f'Test labels = {labels}')
            self.ifca_val_mapping = region_to_val_data
            self.ifca_test_mapping = region_to_test_data

        return mapping

    def test_global_model(self, validation = True):
        if self.algorithm == 'ifca':
            loss, accuracy = self.__validate_clustered(validation)
        else:
            loss, accuracy = self.__validate_centralized(validation)
        return loss, accuracy

    def __validate_centralized(self, validation):
        model = self.server.model
        if validation:
            dataset = self.validation_data
        else:
            dataset = self.test_data
        loss, accuracy = utils.test_model(model, dataset, self.batch_size, self.device)
        if not validation:
            data = pd.DataFrame({'Loss': [loss], 'Accuracy': [accuracy]})
            data.to_csv(f'{self.export_path}-test.csv', index=False)
        return loss, accuracy

    def __validate_clustered(self, validation):
        # models = self.server.model
        if validation:
            mapping = self.ifca_val_mapping
        else:
            mapping = self.ifca_test_mapping

        losses = []
        accuracies = []
        for client in self.clients:
            cluster_id, model = client.model
            loss, accuracy = utils.test_model(model, mapping[cluster_id], self.batch_size, self.device)
            losses.append(loss)
            accuracies.append(accuracy)
        loss, accuracy = sum(losses)/len(losses), sum(accuracies)/len(accuracies)

        if not validation:
            data = pd.DataFrame({'Loss': [loss], 'Accuracy': [accuracy]})
            data.to_csv(f'{self.export_path}-test.csv', index=False)

        return loss, accuracy

    def export_data(self, global_round, training_loss, evaluation_loss, evaluation_accuracy):
        """
        Registers new data, you can use it at each time stamp to store training and evaluation data.
        Important: it does not save the data on a file, you must call the specific method at the end of the simulation!
        :return: Nothing
        """
        self.simulation_data = self.simulation_data._append(
            {'Round': global_round,'TrainingLoss': training_loss, 'ValidationLoss': evaluation_loss, 'ValidationAccuracy': evaluation_accuracy},
            ignore_index=True
        )

    def save_data(self):
        """
        Saves the registered data on a file.
        :return: Nothing
        """
        self.simulation_data.to_csv(f'{self.export_path}.csv', index=False)