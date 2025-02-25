"""Train mbi models."""
# import clize
# import keras
# import keras.losses
import numpy as np
from glob import glob
import os
import matplotlib

from models import Wave_net

if os.uname().nodename == 'france-XPS':
    matplotlib.use('TkAgg')
    basedir = '/home/france/Mounted_dir'
else:
    matplotlib.use('Agg')
    basedir = '/projects/ag-bozek/france'
import matplotlib.pyplot as plt
import torch
from scipy.io import savemat
from time import time
from utils import load_dataset, get_ids, create_run_folders
import logging
import json
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from DISK.utils.utils import read_constant_file
from preprocess_data import preprocess_data, z_score_data, apply_z_score


def _disk_loader(filepath, input_length=9, output_length=1, stride=1, middle_point='', front_point=''):
    """Load keypoints from DISK already processed .npz files."""

    dataset_constant_file = glob(os.path.join(os.path.dirname(filepath), 'constants.py'))[0]
    dataset_constants = read_constant_file(dataset_constant_file)

    data = np.load(filepath)

    bodyparts = dataset_constants.KEYPOINTS
    coords = data['X']
    if 'time' in data:
        logging.info(f'[testing2/_disk_loader] Input file has time key')
        time_ = (data['time'] * dataset_constants.FREQ).astype(int)
        new_time = np.array([np.arange(np.max(time_) + 1) for _ in range(coords.shape[0])])
        new_coords = np.zeros((coords.shape[0], np.max(time_) + 1, coords.shape[2]), dtype=coords.dtype) * np.nan
        for i in range(len(time_)):
            # print(new_coords[i].shape, (time_[i][time_[i] >= 0]).dtype, (time_[i][time_[i] >= 0]).shape, coords[i].shape)
            new_coords[i, time_[i][time_[i] >= 0]] = coords[i][time_[i] >= 0]
    else:
        new_coords = coords

    exclude_value = np.nan

    # here we can preprocess the data without precautions, because no missing data in training scenario
    transformed_coords, rot_angle, mean_position = preprocess_data(new_coords, bodyparts,
                                                                    middle_point=middle_point,
                                                                    front_point=front_point,
                                                                    exclude_value=exclude_value)

    _, marker_means, marker_stds = z_score_data(transformed_coords.reshape(1, -1, transformed_coords.shape[2]),
                                                exclude_value=exclude_value)

    z_score_coords = apply_z_score(transformed_coords, marker_means, marker_stds, exclude_value)
    # marker_means = None
    # marker_stds = None

    # unproc_X = unprocess_data(z_score_coords, rot_angle, mean_position, marker_means, marker_stds, bodyparts, exclude_value)
    #
    # items = np.random.choice(transformed_coords.shape[0], 1)
    # for item in items:
    #     fig, axes = plt.subplots(transformed_coords.shape[-1]//3, 3, figsize=(10, 10))
    #     axes = axes.flatten()
    #     for i in range(transformed_coords.shape[-1]):
    #         x = new_coords[item, :, i]
    #         x[get_mask(x, exclude_value)] = np.nan
    #         axes[i].plot(x, 'o-')
    #
    #         x = unproc_X[item, :, i]
    #         x[get_mask(x, exclude_value)] = np.nan
    #         axes[i].plot(x, 'o-')


    ## reshape the data to match input_length, output_length
    idx = np.arange(0, z_score_coords.shape[1] - (input_length + output_length), stride)
    input_coords = np.vstack([[v[i: i + input_length][np.newaxis] for i in idx] for v in z_score_coords])
    input_coords = input_coords.reshape((input_coords.shape[0], input_length, len(bodyparts), -1))[..., :dataset_constants.DIVIDER].reshape((input_coords.shape[0], input_length, -1))

    output_coords = np.vstack([[v[i + input_length: i + input_length + output_length][np.newaxis] for i in idx] for v in z_score_coords])
    output_coords = output_coords.reshape((output_coords.shape[0], output_length, len(bodyparts), -1))[..., :dataset_constants.DIVIDER].reshape((output_coords.shape[0], output_length, -1))

    return input_coords.astype(np.float32), output_coords.astype(np.float32), dataset_constants


class CustomDataset(Dataset):
    def __init__(self, X, y):
        super(Dataset, self).__init__()
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        return self.X[item], self.y[item]


def testing_single_model_like_training(val_file, *, front_point='', middle_point='',
          model_name,
          net_name="wave_net", clean=False, input_length=9,
          output_length=1,  stride=1,
          batch_size=1000,
          lossfunc='mean_squared_error',
          device=torch.device('cpu')):
    """Trains the network and saves the results to an output directory.

    :param data_path: Path to an HDF5 file with marker data.
    :param base_output_path: Path to folder in which the run data folder will
                             be saved
    :param run_name: Name of the training run. If not specified, will be
                     formatted according to other parameters.
    :param data_name: Name of the dataset for use in formatting run_name
    :param net_name: Name of the network for use in formatting run_name
    :param clean: If True, deletes the contents of the run output path
    :param input_length: Number of frames to input into model
    :param output_length: Number of frames model will attempt to predict
    :param stride: Downsampling rate of training set.

    """

    # Load Data
    logging.info('Loading Data')
    val_input, val_output, dataset_constants = _disk_loader(val_file, input_length=input_length,
                                                            output_length=output_length, stride=stride,
                                                            middle_point=middle_point, front_point=front_point)
    val_dataset = CustomDataset(val_input, val_output)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)
    logging.info(f'Data loaded, number of val samples {len(val_dataset)}')

    # Create network
    logging.info('Compiling network')
    model = None
    if isinstance(net_name, torch.nn.Module):
        model = net_name
    elif net_name == 'wave_net':

        if model is None:
            logging.info(
                f'Loading ensemble model from {os.path.join(os.path.dirname(model_name), "training_info.json")}')
            with open(os.path.join(os.path.dirname(model_name), "training_info.json"), 'r') as fp:
                dict_training = json.load(fp)
            model = Wave_net(device=device, **dict_training).to(device)
            checkpoint = torch.load(os.path.join(basedir, model_name))
            if 'model_state_dict' in checkpoint.keys():
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            model.eval()

    run_path = os.path.dirname(model_name)

    # params you need to specify:
    if lossfunc == 'mean_squared_error':
        loss_function = torch.nn.MSELoss()

    val_batches = len(val_loader)

    # ----------------- VALIDATION ONLY  -----------------
    with torch.no_grad():
        val_loss = 0
        list_y = []
        val_outputs = []
        for i, data in enumerate(val_loader):
            X = data[0].to(device)
            y = data[1].to(device)

            outputs = model(X) # this gets the prediction from the network
            val_outputs.append(outputs.cpu().numpy())

            val_loss += loss_function(outputs, y).item() # MODIFIED - added [][]
            list_y.append(y.cpu().numpy())

        logging.info(f"Validation loss: {val_loss/val_batches:.4f}")

    array_y = np.vstack(list_y)
    array_outputs = np.vstack(val_outputs)
    rmse = np.squeeze(np.sqrt((array_y - array_outputs)**2))
    rmse = np.nanmean(rmse, axis=1)

    # plot some examples of prediction
    for item in np.random.randint(0, X.shape[0], 10):
        fig, axes = plt.subplots(8, dataset_constants.DIVIDER, figsize=(10, 10), sharey='col')
        axes = axes.flatten()
        for i in range(dataset_constants.NUM_FEATURES):
            axes[i].plot(X.detach().cpu().numpy()[item, :, i], 'o-')
            axes[i].plot([9], list_y[-1][item, :, i], 'o')
            axes[i].plot([9], val_outputs[-1][item, :, i], 'x')

        plt.suptitle(f'RMSE = {rmse[item]:.3f}')
        plt.savefig(os.path.join(run_path, f'testing_like_training_last_epoch_prediction_val_item-{item}.png'))
        plt.close()

    # plot RMSE histogram
    plt.figure()
    plt.hist(rmse, bins=50)
    plt.yscale('log')
    plt.suptitle(f'mean RMSE: {np.mean(rmse):.3f} +/- {np.std(rmse):.3f}')
    plt.savefig(os.path.join(run_path, f'testing_like_training_last_epoch_hist_val_RMSE.png'))
    plt.close()

