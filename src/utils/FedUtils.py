import torch
import torch.nn as nn
import seaborn as sns
from models.MNIST import NNMnist
import matplotlib.pyplot as plt
import torch.nn.utils.prune as tprune
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2

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
    criterion = nn.CrossEntropyLoss()
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
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

def prune_model(model_params, dataset_name, amount):
    model = initialize_model(dataset_name)
    model.load_state_dict(model_params)
    # Pruning
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            tprune.l1_unstructured(module, name='weight', amount=amount)

    #Remove the pruning reparametrizations to make the model explicitly sparse
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear):
            tprune.remove(module, 'weight')
    return model.state_dict()


def check_sparsity(state_dict, verbose=False):
    total_zeros = 0
    total_params = 0

    for name, tensor in state_dict.items():

        num_params = tensor.numel()
        num_zeros = torch.sum(tensor == 0).item()

        total_params += num_params
        total_zeros += num_zeros

        if verbose:
            layer_sparsity = (num_zeros / num_params) * 100
            print(f"Layer: {name} | Sparsity: {layer_sparsity:.2f}%")

    if total_params == 0:
        return 0.0

    global_sparsity = (total_zeros / total_params) * 100
    return global_sparsity