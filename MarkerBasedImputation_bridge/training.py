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

# def create_model(net_name, **kwargs):
#     """Initialize a network for training."""
#     # compile_model = dict(
#     #     wave_net=models.wave_net,
#     #     lstm_model=models.lstm_model,
#     #     wave_net_res_skip=models.wave_net_res_skip
#     #     ).get(net_name)
#     compile_model = Wave_net
#     if compile_model is None:
#         return None
#
#     return compile_model(**kwargs)


def _disk_loader(filepath, input_length=9, output_length=1, stride=1, middle_point='', front_point=''):
    """
    Load keypoints from .npz files already processed for DISK

    :args filepath: full path to npz file (str)
    :args input_length: to reshape the input data to fit the NN model input (int, default: 9)
    :args output_length: to reshape the input data to fit the NN model output (int, default: 1)
    :args stride: delta between two `input_length` samples from the original data (int, default: 1)
    :args middle_point: reference point to re-align the data - should be a point matching the keypoint names from the
                        constant file and in the middle of the animal (in DANNCE original data: SpineM)
                        (str or list of str, no default)
    :args front_point: reference point to re-align the data - should be a point matching the keypoint names from the
                        constant file and in the front of the animal (in DANNCE original data: SpineF)
                        (str or list of str, no default)

    :return input_coords: np.array (Float32) with samples, shape (samples, input_length, keypoints * 2 or 3D)
    :return output_coords:  np.array (Float32) with samples, shape (samples, output_length, keypoints * 2 or 3D)
    :return dataset_constants: dict with dataset constants, directly loaded from the constants.pz file
                               (with, e.g., keypoint_names)
    """

    dataset_constant_file = glob(os.path.join(os.path.dirname(filepath), 'constants.py'))[0]
    dataset_constants = read_constant_file(dataset_constant_file)
    divider = dataset_constants.DIVIDER

    data = np.load(filepath)

    bodyparts = dataset_constants.KEYPOINTS
    coords = data['X']
    if 'time' in data:
        logging.info(f'[training/_disk_loader] Input file has time key')
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
    transformed_coords, rot_angle, mean_position = preprocess_data(new_coords, bodyparts, dataset_constants.DIVIDER,
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
    input_coords = input_coords.reshape((input_coords.shape[0], input_length, len(bodyparts), -1))[..., :divider].reshape((input_coords.shape[0], input_length, -1))

    output_coords = np.vstack([[v[i + input_length: i + input_length + output_length][np.newaxis] for i in idx] for v in z_score_coords])
    output_coords = output_coords.reshape((output_coords.shape[0], output_length, len(bodyparts), -1))[..., :divider].reshape((output_coords.shape[0], output_length, -1))

    return input_coords.astype(np.float32), output_coords.astype(np.float32), dataset_constants


class CustomDataset(Dataset):
    """Ultra-simple Dataset class to hand the data to pytorch DataLoader"""
    def __init__(self, X, y):
        super(Dataset, self).__init__()
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        return self.X[item], self.y[item]


def train(train_file, val_file, *, front_point='', middle_point='',
          base_output_path="models", run_name=None,
          data_name=None, net_name="wave_net", clean=False, input_length=9,
          output_length=1,  stride=1, train_fraction=.85,
          val_fraction=0.15, only_moving_frames=False, n_filters=512,
          filter_width=2, layers_per_level=3, n_dilations=None,
          latent_dim=750, epochs=50, batch_size=1000,
          lossfunc='mean_squared_error', lr=1e-4, batches_per_epoch=0,
          val_batches_per_epoch=0, reduce_lr_factor=0.5, reduce_lr_patience=3,
          reduce_lr_min_delta=1e-5, reduce_lr_cooldown=0,
          reduce_lr_min_lr=1e-10, save_every_epoch=False,
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
    :param n_markers: Number of markers to use
    :param stride: Downsampling rate of training set.
    :param train_fraction: Fraction of dataset to use as training
    :param val_fraction: Fraction of dataset to use as validation
    :param only_moving_frames: If True only use moving_frames.
    :param filter_width: Width of base convolution filter
    :param layers_per_level: Number of layers to use at each convolutional
                             block
    :param n_dilations: Number of dilations for wavenet filters.
                        (See models.wave_net)
    :param latent_dim: Number of latent dimensions (Currently just for LSTM)
    :param n_filters: Number of filters to use as baseline (see create_model)
    :param epochs: Number of epochs to train for
    :param batch_size: Number of samples per batch
    :param batches_per_epoch: Number of batches per epoch (validation is
                              evaluated at the end of the epoch)
    :param val_batches_per_epoch: Number of batches for validation
    :param reduce_lr_factor: Factor to reduce the learning rate by (see
                             ReduceLROnPlateau)
    :param reduce_lr_patience: How many epochs to wait before reduction (see
                               ReduceLROnPlateau)
    :param reduce_lr_min_delta: Minimum change in error required before
                                reducing LR (see ReduceLROnPlateau)
    :param reduce_lr_cooldown: How many epochs to wait after reduction before
                               LR can be reduced again (see ReduceLROnPlateau)
    :param reduce_lr_min_lr: Minimum that the LR can be reduced down to (see
                             ReduceLROnPlateau)
    :param save_every_epoch: Save weights at every epoch. If False, saves only
                             initial, final and best weights.
    """
    start_ts = time()

    # Set the n_dilations param
    if n_dilations is None:
        n_dilations = int(np.floor(np.log2(input_length)))
    else:
        n_dilations = int(n_dilations)

    # Load Data
    logging.info('Loading Data')

    train_input, train_output, dataset_constants = _disk_loader(train_file, input_length=input_length, output_length=output_length,
                                                                middle_point=middle_point, front_point=front_point)
    val_input, val_output, _ = _disk_loader(val_file, input_length=input_length, output_length=output_length,
                                            stride=stride, middle_point=middle_point, front_point=front_point)

    train_dataset = CustomDataset(train_input, train_output)
    val_dataset = CustomDataset(val_input, val_output)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)
    logging.info(f'Data loaded, number of train samples {len(train_dataset)}, number of val samples {len(val_dataset)}')

    # Create network
    logging.info('Compiling network')
    model = None
    if isinstance(net_name, torch.nn.Module):
        model = net_name
        net_name = model.name
    elif net_name == 'wave_net':
        model = Wave_net(input_length=input_length,
                             output_length=output_length, n_markers=dataset_constants.N_KEYPOINTS * dataset_constants.DIVIDER,
                             n_filters=n_filters, filter_width=filter_width,
                             layers_per_level=layers_per_level, device=device,
                             n_dilations=n_dilations, print_summary=False).to(device)
    # FR: not implementing the other models (lstm and wave_net_res_skip)
    # elif net_name == 'lstm_model':
    #     model = create_model(net_name, lossfunc=lossfunc, lr=lr,
    #                          input_length=input_length, n_markers=n_markers,
    #                          latent_dim=latent_dim, print_summary=False)
    # elif net_name == 'wave_net_res_skip':
    #     model = create_model(net_name, lossfunc=lossfunc, lr=lr,
    #                          input_length=input_length, n_markers=n_markers,
    #                          n_filters=n_filters, filter_width=filter_width,
    #                          layers_per_level=layers_per_level,
    #                          n_dilations=n_dilations, print_summary=True)
    if model is None:
        logging.info(f"Could not find model: {net_name}")
        return

    # Build run name if needed
    if data_name is None:
        data_name = os.path.splitext(base_output_path)[0]
    if run_name is None:
        run_name = "%s-%s_epochs=%d_input_%d_output_%d" % (data_name, net_name, epochs, input_length, output_length)
    logging.info(f"data_name: {data_name}")
    logging.info(f"run_name: {run_name}")

    # Initialize run directories
    logging.info('Building run folders')
    run_path = create_run_folders(run_name, base_path=base_output_path, clean=clean)

    # Save the training information in a mat file.
    logging.info(f'Saving training info in {os.path.join(run_path, "training_info.json")}')
    with open(os.path.join(run_path, "training_info.json"), "w") as fp:
        json.dump({"data_path": train_file[len(basedir):].lstrip('/'),
                        "base_output_path": base_output_path[len(basedir):].lstrip('/'),
                        "run_name": run_name,
                        "data_name": data_name,
                        "net_name": net_name,
                        "clean": clean,
                        "stride": stride,
                        "input_length": input_length,
                        "output_length": output_length,
                        "n_filters": n_filters,
                        "n_markers": dataset_constants.N_KEYPOINTS * dataset_constants.DIVIDER,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "train_fraction": train_fraction,
                        "val_fraction": val_fraction,
                        "only_moving_frames": only_moving_frames,
                        "filter_width": filter_width,
                        "layers_per_level": layers_per_level,
                        "n_dilations": n_dilations,
                        "batches_per_epoch": batches_per_epoch,
                        "val_batches_per_epoch": val_batches_per_epoch,
                        "reduce_lr_factor": reduce_lr_factor,
                        "reduce_lr_patience": reduce_lr_patience,
                        "reduce_lr_min_delta": reduce_lr_min_delta,
                        "reduce_lr_cooldown": reduce_lr_cooldown,
                        "reduce_lr_min_lr": reduce_lr_min_lr,
                        "save_every_epoch": save_every_epoch}, fp)

    print_every = 1
    if lossfunc == 'mean_squared_error':
        loss_function = torch.nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', min_lr=reduce_lr_min_lr,
                                                    cooldown=reduce_lr_cooldown, eps=reduce_lr_min_delta,
                                                    patience=reduce_lr_patience, factor=reduce_lr_factor,
                                                    verbose=True)
    # from keras: (monitor="val_loss", factor=reduce_lr_factor, patience=reduce_lr_patience, verbose=1, mode="auto",
    # epsilon=reduce_lr_min_delta, cooldown=reduce_lr_cooldown, min_lr=reduce_lr_min_lr)

    losses = []
    val_losses = []

    batches = len(train_loader)
    val_batches = len(val_loader)
    best_val_loss = np.inf
    # loop for every epoch (training + evaluation)
    for epoch in range(epochs):
        total_loss = 0

        # ----------------- TRAINING  --------------------
        model.train()
        list_y_train = []

        for i, data in enumerate(train_loader):
            model.zero_grad() # to make sure that all the grads are 0
            X = data[0].to(device)
            y = data[1].to(device)
            outputs = model(X) # forward
            if torch.isnan(X).any():
                print(torch.where(torch.isnan(X)), outputs, y, flush=True)
            loss = loss_function(outputs, y)

            list_y_train.append(y.detach().cpu().numpy())

            loss.backward() # accumulates the gradient (by addition) for each parameter.
            optimizer.step() # performs a parameter update based on the current gradient

            # getting training quality data
            current_loss = loss.item()
            total_loss += current_loss

        # ----------------- VALIDATION  -----------------
        if epoch >= 0 and (epoch % print_every == 0 or epoch == epochs - 1):

            # set model to evaluating (testing)
            model.eval()
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
            scheduler.step(val_loss)

            logging.debug(f'{np.unique(np.vstack(val_outputs).flatten())}')
            logging.info(f"Epoch {epoch+1}/{epochs}, training loss: {total_loss/batches:.4f}, validation loss: {val_loss/val_batches:.4f}")
            losses.append(total_loss/batches) # for plotting learning curve
            val_losses.append(val_loss/val_batches) # for plotting learning curve
            if val_loss/val_batches < best_val_loss:
                best_val_loss = val_loss/val_batches
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': loss,
                        }, os.path.join(run_path, "best_model.h5"))

    array_y = np.vstack(list_y)
    array_outputs = np.vstack(val_outputs)
    rmse = np.squeeze(np.sqrt((array_y - array_outputs)**2))
    rmse = np.nanmean(rmse, axis=1)

    # plot some examples of predictions on the validation set
    items = np.random.randint(0, X.shape[0], 10)
    for item in items:
        fig, axes = plt.subplots(8, dataset_constants.DIVIDER, figsize=(10, 10), sharey='col')
        axes = axes.flatten()
        print(X.shape, dataset_constants.NUM_FEATURES)
        for i in range(dataset_constants.NUM_FEATURES):
            axes[i].plot(X.detach().cpu().numpy()[item, :, i], 'o-')
            axes[i].plot([9], list_y[-1][item, :, i], 'o')
            axes[i].plot([9], val_outputs[-1][item, :, i], 'x')
        plt.suptitle(f'RMSE = {rmse[item]:.3f}')
        plt.savefig(os.path.join(run_path, f'last_epoch_prediction_val_item-{item}.png'))
        plt.close()

    # plot the histogram of RMSEs
    plt.figure()
    plt.hist(rmse, bins=50)
    plt.yscale('log')
    plt.suptitle(f'mean RMSE: {np.mean(rmse):.3f} +/- {np.std(rmse):.3f}')
    plt.savefig(os.path.join(run_path, f'last_epoch_hist_val_RMSE.png'))
    plt.close()

    # plot the learning curves
    fig, ax = plt.subplots(1, 1)
    x = np.arange(0, epochs, print_every)
    ax.plot(x, losses, label='train')
    ax.plot(x, val_losses, '--', label='val')
    ax.legend()
    ax.set_ylabel('Loss')
    plt.savefig(os.path.join(run_path, f'loss_curves.png'))
    plt.close()

    logging.info(f"Training time: {time()-start_ts}s")


if __name__ == '__main__':

    BASEFOLDER = os.path.join(basedir, "results_behavior/MarkerBasedImputation")
    DATASETPATH = os.path.join(basedir, 'results_behavior/datasets/INH_FL2_keypoints_1_60_wresiduals_w1nan_stride0.5_new')
    train_file = os.path.join(DATASETPATH, 'train_dataset_w-0-nans.npz')
    val_file = os.path.join(DATASETPATH, 'val_dataset_w-0-nans.npz')
    MODELFOLDER = os.path.join(BASEFOLDER, "models")
    PREDICTIONSPATH = os.path.join(BASEFOLDER, "predictions")
    MERGEDFILE = os.path.join(BASEFOLDER, "/fullDay_model_ensemble.h5")

    # Training
    NMODELS = 1
    TRAINSTRIDE = 1 #5 # FL2 is a smaller dataset than they had (25 million frames for training)
    EPOCHS = 30

    # # Imputation
    # NFOLDS = 20
    # IMPUTESTRIDE = 5
    # ERRORDIFFTHRESH = .5

    # $1 - -base - output - path =$2 - -epochs =$3 - -stride =$4
    # $DATASETPATH $MODELBASEOUTPUTPATH $EPOCHS $TRAINSTRIDE
    device = torch.device('cuda:0')

    for _ in range(NMODELS):
        train(train_file, val_file, base_output_path=MODELFOLDER, run_name=None,
              data_name=None, net_name="wave_net", clean=False, input_length=9,
              output_length=1, stride=TRAINSTRIDE, train_fraction=.85,
              val_fraction=0.15, only_moving_frames=False, n_filters=512,
              filter_width=2, layers_per_level=3, n_dilations=None,
              latent_dim=750, epochs=EPOCHS, batch_size=1000,
              lossfunc='mean_squared_error', lr=1e-4, batches_per_epoch=0,
              val_batches_per_epoch=0, reduce_lr_factor=0.5, reduce_lr_patience=3,
              reduce_lr_min_delta=1e-5, reduce_lr_cooldown=0,
              reduce_lr_min_lr=1e-10, save_every_epoch=False, device=device)