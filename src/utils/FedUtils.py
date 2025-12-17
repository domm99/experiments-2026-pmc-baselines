import math
import random
import torch
import numpy as np
import torch.nn as nn
import seaborn as sns
from models.MNIST import NNMnist
import matplotlib.pyplot as plt
from torchvision.models import mobilenet_v2
from torch.utils.data import Dataset, Subset, DataLoader

def initialize_model(name):
    if name == 'MNIST' or name == 'FashionMNIST':
        return NNMnist()
    elif name == 'EMNIST':
        return NNMnist(output_size=27)
    elif name == 'UTKFace':
        return NNMnist(output_size=1) # TODO - fix
    elif name == 'CIFAR100':
        model = mobilenet_v2(pretrained=True)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, 100)
        )
        return model
    else:
        raise Exception(f'Model {name} not implemented! Please check :)')

def initialize_control_state(experiment, device):
    control_state = initialize_model(experiment).to(device)
    for param in control_state.parameters():
        nn.init.constant_(param, 0.0)
    return control_state.state_dict()

def test_model(model, dataset, batch_size, device):
    criterion = nn.NLLLoss()
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for batch_index, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
    accuracy = correct / total
    return loss, accuracy

def plot_heatmap(data, labels, areas, name, floating = True):
    sns.heatmap(data, annot=True, cmap="YlGnBu",
                xticklabels=[f'{i}' for i in range(labels)],
                yticklabels=[f'{i}' for i in range(areas)],
                fmt= '.3f' if floating else 'd'
                )
    plt.xlabel('Label')
    plt.ylabel('Area')
    plt.tight_layout()
    plt.savefig(f'{name}.pdf')
    plt.close()