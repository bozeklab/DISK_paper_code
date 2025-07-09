import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import importlib.util
import plotly.graph_objects as go

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn.functional as F

from DISK.utils.dataset_utils import SupervisedDataset, FullLengthDataset


def plot_save(plot_fct, save_bool=True, title='', only_png=False, outputdir=''):
    with sns.axes_style("ticks"):  # plt.style.context('dark_background'):
        sns.despine()
        plot_fct()
        if save_bool:
            if only_png:
                plt.savefig(os.path.join(outputdir, title + '_dark.png'), transparent=True)
            else:
                plt.savefig(os.path.join(outputdir, title + '.svg'))
            plt.close()
    with plt.style.context('seaborn'):
        plot_fct()
        if save_bool:
            plt.savefig(os.path.join(outputdir, title + '.png'))
            plt.close()


def read_constant_file(constant_file):
    """import constant file as a python file from its path"""
    spec = importlib.util.spec_from_file_location("module.name", constant_file)
    constants = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(constants)

    try:
        constants.NUM_FEATURES, constants.DIVIDER, constants.KEYPOINTS, constants.SEQ_LENGTH
    except NameError:
        print('constant file should have following keys: NUM_FEATURES, DIVIDER, KEYPOINTS, SEQ_LENGTH')
    constants.N_KEYPOINTS = len(constants.KEYPOINTS)

    return constants


def plot_training(df, offset=10):
    fig, axes = plt.subplots(2, 1, sharex='all')
    axes[0].plot(df[0][offset:], label='train')
    axes[0].plot(df[2][offset:], label='validation')
    axes[0].legend()
    axes[0].set_title('Loss')
    axes[1].plot(df[1][offset:])
    axes[1].plot(df[3][offset:])
    axes[1].set_title('RMSE')


def plot_training_full(df, offset=10):
    fig, axes = plt.subplots(6, 1, sharex='all', figsize=(10, 6 * 3))
    axes[0].plot(df[0][offset:], label='train')
    axes[0].plot(df[2][offset:], label='validation')
    axes[0].legend()
    axes[0].set_title('Total Loss')
    axes[1].plot(df[5][offset:])
    axes[1].plot(df[9][offset:])
    axes[1].set_title('Original loss')
    axes[2].plot(df[6][offset:])
    axes[2].plot(df[10][offset:])
    axes[2].set_title('Mu sigma loss')
    axes[3].plot(df[7][offset:])
    axes[3].plot(df[11][offset:])
    axes[3].set_title('Smooth loss')
    axes[4].plot(df[8][offset:])
    axes[4].plot(df[12][offset:])
    axes[4].set_title('Uncertainty loss')
    axes[5].plot(df[1][offset:])
    axes[5].plot(df[3][offset:])
    axes[5].set_title('RMSE')


def plot_history(history, every):
    fig, axes = plt.subplots(2, 1, sharex='all')
    x_train = np.arange(len(history['loss']))
    x_val = np.arange(0, len(history['loss']), every)
    pl_train = axes[0].plot(x_train, history['loss'])
    pl_val = axes[0].plot(x_val, history['val_loss'])
    axes[0].set_title('Loss')
    axes[1].plot(x_train, history['accuracy'], c=pl_train[0].get_color())
    axes[1].plot(x_val, history['val_accuracy'], c=pl_val[0].get_color())
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epochs')
    # plt.show()


def compute_accuracy(output, labels):
    ## Calculating the accuracy
    # Model's output is log-softmax, take exponential to get the probabilities
    ps = torch.exp(output)
    # Class with highest probability is our predicted class, compare with true label
    equality = (labels.data == ps.max(1)[1])
    # Accuracy is number of correct predictions divided by all predictions, just take the mean
    return equality.type_as(torch.FloatTensor()).mean()


def compute_confusion_matrix(output, labels, n_classes):
    pred = torch.argmax(output, dim=1).detach().cpu().numpy()
    gt = labels.detach().cpu().numpy()
    # print('--', np.unique(pred, return_counts=True), np.unique(gt, return_counts=True))
    return confusion_matrix(gt, pred, labels=list(range(n_classes)))


def validation(model, testloader, criterion, n_classes, device='cpu', run_name='mlstm-fcn'):
    accuracy = 0
    test_loss = 0
    conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
    for inputs, _, labels, seq_lens, _, _, _ in testloader:
        inputs = inputs.float()
        inputs, labels = inputs.to(device), labels.to(device)

        if run_name == 'mlstm-fcn':
            output = model.forward(inputs, seq_lens)
        elif run_name == 'st-gcn':
            inputs = torch.unsqueeze(
                torch.moveaxis(inputs.view((inputs.shape[0], inputs.shape[1], inputs.shape[2] // 3, 3)), -1, 1), -1)
            output = model.forward(inputs)
            output = F.log_softmax(output, dim=1)
        else:
            inputs = torch.unsqueeze(
                torch.moveaxis(inputs.view((inputs.shape[0], inputs.shape[1], inputs.shape[2] // 3, 3)), -1, 1), -1)
            output = model.forward(inputs)

        labels = labels.flatten().type(torch.long)  # type for the NLL loss

        test_loss += criterion(output, labels).item()
        conf_matrix += compute_confusion_matrix(output, labels, n_classes)
        accuracy += compute_accuracy(output, labels)

    return test_loss, accuracy, conf_matrix


def train(model, trainloader, validloader, criterion, optimizer, scheduler,
          epochs=10, n_classes=3, device='cpu', run_name='mlstm-fcn', every=10):
    print("Training started on device: {}".format(device))

    history = {'accuracy': [],
               'loss': [],
               'val_accuracy': [],
               'val_loss': []}
    valid_loss_min = np.Inf  # track change in validation loss

    train_accuracy = 0.
    train_loss = 0.0
    train_conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
    last_train_conf_matrix = np.array(train_conf_matrix)
    last_val_conf_matrix = np.array(train_conf_matrix)

    for e in range(epochs):
        model.train()
        for inputs, _, labels, seq_lens, _, _, _ in trainloader:

            inputs = inputs.float()
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.flatten().type(torch.long)  # type for the NLL loss

            optimizer.zero_grad()

            if run_name == 'mlstm-fcn':
                output = model.forward(inputs, seq_lens)
            elif run_name == 'st-gcn':
                inputs = torch.unsqueeze(
                    torch.moveaxis(inputs.view((inputs.shape[0], inputs.shape[1], inputs.shape[2] // 3, 3)), -1, 1), -1)
                output = model.forward(inputs)
                output = F.log_softmax(output, dim=1)
            else:
                inputs = torch.unsqueeze(
                    torch.moveaxis(inputs.view((inputs.shape[0], inputs.shape[1], inputs.shape[2] // 3, 3)), -1, 1), -1)
                output = model.forward(inputs)

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            acc = compute_accuracy(output, labels)
            train_conf_matrix += compute_confusion_matrix(output, labels, n_classes)
            train_accuracy = train_accuracy + acc

        scheduler.step()

        model.eval()
        print("Epoch: {}/{}.. ".format(e + 1, epochs),
              "Training Loss: {:.6f}.. ".format(train_loss / len(trainloader)),
              "Training Accuracy: {:.2f}%  --  ".format((train_accuracy / len(trainloader)) * 100))
        history['loss'].append(train_loss / len(trainloader))
        history['accuracy'].append((train_accuracy / len(trainloader)) * 100)

        if (e + 1) % every == 0:
            with torch.no_grad():
                valid_loss, accuracy, conf_matrix = validation(model, validloader, criterion, n_classes, device,
                                                               run_name)

            print(" " * 14,
                  "Val Loss: {:.6f}.. ".format(valid_loss / len(validloader)),
                  "Val Accuracy: {:.2f}%".format(accuracy / len(validloader) * 100))
            last_train_conf_matrix = np.array(train_conf_matrix)
            last_val_conf_matrix = np.array(conf_matrix)

            print('TRAIN:', train_conf_matrix)
            print('VAL:', conf_matrix)

            history['val_loss'].append(valid_loss / len(validloader))
            history['val_accuracy'].append(accuracy / len(validloader) * 100)

            # save model if validation loss has decreased
            if valid_loss / len(validloader) <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    valid_loss_min,
                    valid_loss / len(validloader)))
                torch.save(model.state_dict(), 'weights/' + run_name + '.pt')
                valid_loss_min = valid_loss / len(validloader)

        train_loss = 0
        train_accuracy = 0
        train_conf_matrix = np.zeros((n_classes, n_classes), dtype=int)

        # model.train()
    return history, last_train_conf_matrix, last_val_conf_matrix


def load_datasets(dataset_name='ISLD', dataset_type='supervised', root_path='', suffix='', **kwargs):
    """
    Folder structure: all pt files in the same subfolder in datasets

    X: (batch size, time, channels)
    y: (batch_size) with integer class
    lens: (batch_size) gives the input sequence length, the rest is filled with 0

    3 datasets: train, validation and test

    :param dataset_name: subfolder name
    :return: 3 torch datasets (train_dataset, val_dataset, test_dataset)
    """


    data_path = os.path.join(root_path, 'datasets', dataset_name)

    if dataset_type == 'supervised':
        train_dataset = SupervisedDataset(os.path.join(data_path, f'train_dataset{suffix}.npz'), **kwargs)
        test_dataset = SupervisedDataset(os.path.join(data_path, f'test_dataset{suffix}.npz'), **kwargs)
        val_dataset = SupervisedDataset(os.path.join(data_path, f'val_dataset{suffix}.npz'), **kwargs)
    elif dataset_type == 'full_length':
        train_dataset = FullLengthDataset(os.path.join(data_path, f'train_fulllength_dataset{suffix}.npz'),
                                          **kwargs)
        test_dataset = FullLengthDataset(os.path.join(data_path, f'test_fulllength_dataset{suffix}.npz'), **kwargs)
        val_dataset = FullLengthDataset(os.path.join(data_path, f'val_fulllength_dataset{suffix}.npz'), **kwargs)
    else:
        raise ValueError(f'[load_datasets function] argument dataset_type = {dataset_type} is not recognized. '
                         f'Authorized values are "supervised", "self_supervised", "full_length"')

    return train_dataset, val_dataset, test_dataset


def read_constant_file(constant_file):
    ## import constant file as a python file from its path
    spec = importlib.util.spec_from_file_location("module.name", constant_file)
    constants = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(constants)

    try:
        constants.W_RESIDUALS, constants.NUM_FEATURES, constants.DIVIDER, constants.SKELETON, constants.KEYPOINTS, constants.MAX_SEQ_LEN
    except NameError:
        print(
            'constant file should have following keys: W_RESIDUALS, NUM_FEATURES, DIVIDER, SKELETON, KEYPOINTS, MAX_SEQ_LEN')
    constants.N_KEYPOINTS = len(constants.KEYPOINTS)

    return constants


class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, total_epoch, after_scheduler=None):
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = -1
        super().__init__(optimizer)

    def get_lr(self):
        return [base_lr * (self.last_epoch + 1) / self.total_epoch for base_lr in self.base_lrs]

    def step(self, epoch=None, metric=None):
        if self.last_epoch >= self.total_epoch - 1:
            if metric is None:
                return self.after_scheduler.step(epoch)
            else:
                return self.after_scheduler.step(metric, epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


def check_data_skeleton_compatibility(dataset_folder):
    data = np.load(os.path.join(dataset_folder, 'test_dataset_w-0-nans.npz'))
    spec = importlib.util.spec_from_file_location("module.name", os.path.join(dataset_folder, 'skeleton.py'))
    skeleton_inputs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(skeleton_inputs)

    n_kp = skeleton_inputs.num_keypoints
    keypoints = skeleton_inputs.keypoints
    keypoints_drawn = {k: False for k in keypoints}
    neighbor_links = skeleton_inputs.neighbor_links

    x = data['X'][0][0]
    x = x.reshape(n_kp, -1)[..., :3]
    print(x.shape, n_kp, keypoints)

    if x.shape[-1] == 3:
        # 3D
        ax = plt.figure(figsize=(10, 10)).add_subplot(projection='3d')
        for list_ in neighbor_links:
            if type(list_[0]) == int:
                if not keypoints_drawn[keypoints[list_[0]]]:
                    plt.plot(x[list_[0], 0], x[list_[0], 1], x[list_[0], 2], 'o', label=f'{list_[0]} {keypoints[list_[0]]}')
                    keypoints_drawn[keypoints[list_[0]]] = True
                if not keypoints_drawn[keypoints[list_[1]]]:
                    plt.plot(x[list_[1], 0], x[list_[1], 1], x[list_[1], 2], 'o', label=f'{list_[1]} {keypoints[list_[1]]}')
                    keypoints_drawn[keypoints[list_[1]]] = True
                plt.plot([x[list_[0], 0], x[list_[1], 0]],
                               [x[list_[0], 1], x[list_[1], 1]],
                               [x[list_[0], 2], x[list_[1], 2]],
                         'k')
            else:
                for pair in list_:
                    if not keypoints_drawn[keypoints[pair[0]]]:
                        plt.plot(x[pair[0], 0], x[pair[0], 1], x[pair[0], 2], 'o', label=f'{pair[0]} {keypoints[pair[0]]}')
                        keypoints_drawn[keypoints[pair[0]]] = True
                    if not keypoints_drawn[keypoints[pair[1]]]:
                        plt.plot(x[pair[1], 0], x[pair[1], 1], x[pair[1], 2], 'o', label=f'{pair[1]} {keypoints[pair[1]]}')
                        keypoints_drawn[keypoints[pair[1]]] = True
                    plt.plot([x[pair[0], 0], x[pair[1], 0]],
                                   [x[pair[0], 1], x[pair[1], 1]],
                                   [x[pair[0], 2], x[pair[1], 2]], 'k')
    else:
        # 2D
        plt.figure(figsize=(10, 10))
        for list_ in neighbor_links:
            if type(list_[0]) == int:
                if not keypoints_drawn[keypoints[list_[0]]]:
                    plt.plot(x[list_[0], 0], x[list_[0], 1], 'o', label=f'{list_[0]} {keypoints[list_[0]]}')
                    keypoints_drawn[keypoints[list_[0]]] = True
                if not keypoints_drawn[keypoints[list_[1]]]:
                    plt.plot(x[list_[1], 0], x[list_[1], 1], 'o', label=f'{list_[1]} {keypoints[list_[1]]}')
                    keypoints_drawn[keypoints[list_[0]]] = True
                plt.plot([x[list_[0], 0], x[list_[1], 0]],
                               [x[list_[0], 1], x[list_[1], 1]], 'k')
            else:
                for pair in list_:
                    if not keypoints_drawn[keypoints[pair[0]]]:
                        plt.plot(x[pair[0], 0], x[pair[0], 1], 'o', label=f'{pair[0]} {keypoints[pair[0]]}')
                        keypoints_drawn[keypoints[pair[0]]] = True
                    if not keypoints_drawn[keypoints[pair[1]]]:
                        plt.plot(x[pair[1], 0], x[pair[1], 1], 'o', label=f'{pair[1]} {keypoints[pair[1]]}')
                        keypoints_drawn[keypoints[pair[1]]] = True
                    plt.plot([x[pair[0], 0], x[pair[1], 0]],
                                   [x[pair[0], 1], x[pair[1], 1]], 'k')
    plt.legend()
    plt.suptitle(os.path.basename(dataset_folder))
    plt.savefig(os.path.join(dataset_folder, f'skeleton_plot_w_kp_names.png'))


