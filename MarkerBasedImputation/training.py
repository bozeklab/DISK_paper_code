"""Train mbi models."""
# import clize
# import keras
# import keras.losses
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from scipy.io import savemat
from time import time
# from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from utils import load_dataset, get_ids, create_run_folders
import models
import json
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


def create_model(net_name, **kwargs):
    """Initialize a network for training."""
    # compile_model = dict(
    #     wave_net=models.wave_net,
    #     lstm_model=models.lstm_model,
    #     wave_net_res_skip=models.wave_net_res_skip
    #     ).get(net_name)
    compile_model = models.wave_net
    if compile_model is None:
        return None

    return compile_model(**kwargs)

class CustomDataset(Dataset):
    def __init__(self, X, y):
        super(Dataset, self).__init__()
        self.X = X
        self.y = y

    def forward(self, item):
        return self.X[item], self.y[item]


def train(data_path, *, base_output_path="models", run_name=None,
          data_name=None, net_name="wave_net", clean=False, input_length=9,
          output_length=1,  n_markers=60, stride=1, train_fraction=.85,
          val_fraction=0.15, only_moving_frames=False, n_filters=512,
          filter_width=2, layers_per_level=3, n_dilations=None,
          latent_dim=750, epochs=50, batch_size=1000,
          lossfunc='mean_squared_error', lr=1e-4, batches_per_epoch=0,
          val_batches_per_epoch=0, reduce_lr_factor=0.5, reduce_lr_patience=3,
          reduce_lr_min_delta=1e-5, reduce_lr_cooldown=0,
          reduce_lr_min_lr=1e-10, save_every_epoch=False,
          device=''):
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
    start_ts = time.time()

    # Set the n_dilations param
    if n_dilations is None:
        n_dilations = np.int32(np.floor(np.log2(input_length)))
    else:
        n_dilations = int(n_dilations)

    # Load Data
    print('Loading Data')
    # markers, marker_means, marker_stds, bad_frames, moving_frames = \
    #     load_dataset(data_path)
    markers, bad_frames = None, None
    # moving_frames = np.squeeze(moving_frames > 0)
    # if only_moving_frames:
    #     markers = markers[moving_frames, :]
    #     bad_frames = bad_frames[moving_frames, :]
    # stride is like a downsampling by picking each n frames
    markers = markers[::stride, :]
    bad_frames = bad_frames[::stride, :]

    # Get Ids
    print('Getting indices')
    [input_ids, target_ids] = get_ids(bad_frames, input_length,
                                      output_length, True, True)

    # Get the training, testing, and validation trajectories by indexing into
    # the marker arrays
    n_train = np.int32(np.round(input_ids.shape[0]*train_fraction))
    n_val = np.int32(np.round(input_ids.shape[0]*val_fraction))
    X = markers[input_ids[:n_train, :], :]
    Y = markers[target_ids[:n_train, :], :]
    val_X = markers[input_ids[n_train:(n_train+n_val), :], :]
    val_Y = markers[target_ids[n_train:(n_train+n_val), :], :]

    train_dataset = CustomDataset(X, Y)
    val_dataset = CustomDataset(val_X, val_Y)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)

    # Create network
    print('Compiling network')
    model = None
    if isinstance(net_name, torch.Module):
        model = net_name
        net_name = model.name
    elif net_name == 'wave_net':
        model = create_model(net_name,
                             input_length=input_length,
                             output_length=output_length, n_markers=n_markers,
                             n_filters=n_filters, filter_width=filter_width,
                             layers_per_level=layers_per_level,
                             n_dilations=n_dilations, print_summary=False)
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
        print("Could not find model:", net_name)
        return

    # Build run name if needed
    if data_name is None:
        data_name = os.path.splitext(os.path.basename(data_path))[0]
    if run_name is None:
        run_name = "%s-%s_epochs=%d_input_%d_output_%d" \
            % (data_name, net_name, epochs, input_length, output_length)
    print("data_name:", data_name)
    print("run_name:", run_name)

    # Initialize run directories
    print('Building run folders')
    run_path = create_run_folders(run_name, base_path=base_output_path,
                                  clean=clean)

    # Save the training information in a mat file.
    print('Saving training info')
    with open(os.path.join(run_path, "training_info"), "w") as fp:
        json.dump({"data_path": data_path, "base_output_path": base_output_path,
             "run_name": run_name, "data_name": data_name,
             "net_name": net_name, "clean": clean, "stride": stride,
             "input_length": input_length, "output_length": output_length,
             "n_filters": n_filters, "n_markers": n_markers, "epochs": epochs,
             "batch_size": batch_size, "train_fraction": train_fraction,
             "val_fraction": val_fraction,
             "only_moving_frames": only_moving_frames,
             "filter_width": filter_width,
             "layers_per_level": layers_per_level, "n_dilations": n_dilations,
             "batches_per_epoch": batches_per_epoch,
             "val_batches_per_epoch": val_batches_per_epoch,
             "reduce_lr_factor": reduce_lr_factor,
             "reduce_lr_patience": reduce_lr_patience,
             "reduce_lr_min_delta": reduce_lr_min_delta,
             "reduce_lr_cooldown": reduce_lr_cooldown,
             "reduce_lr_min_lr": reduce_lr_min_lr,
             "save_every_epoch": save_every_epoch}, fp)



    # params you need to specify:
    print_every = 10
    if lossfunc == 'mean_squared_error':
        loss_function = torch.nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', min_lr=reduce_lr_min_lr,
                                                    cooldown=reduce_lr_cooldown, eps=reduce_lr_min_delta,
                                                    patience=reduce_lr_patience, factor=reduce_lr_factor,
                                                    verbose=True)
    # (monitor="val_loss",
    #                                        factor=reduce_lr_factor,
    #                                        patience=reduce_lr_patience,
    #                                        verbose=1, mode="auto",
    #                                        epsilon=reduce_lr_min_delta,
    #                                        cooldown=reduce_lr_cooldown,
    #                                        min_lr=reduce_lr_min_lr)

    losses = []
    val_losses = []

    batches = len(train_loader)
    val_batches = len(val_loader)

    # loop for every epoch (training + evaluation)
    for epoch in range(epochs):
        total_loss = 0

        # ----------------- TRAINING  --------------------
        model.train()
        list_y_train, list_pred_train = [], []

        for i, data in enumerate(train_loader):
            model.zero_grad() # to make sure that all the grads are 0
            X = data[0].to(device)
            y = data[1].to(device)
            outputs, hist = model(X) # forward
            if torch.isnan(X).any():
                print(torch.where(torch.isnan(X)), outputs, y, flush=True)
            loss = loss_function(torch.clip(outputs, 1e-9, 1-1e-9), y)

            list_y_train.append(y.detach().cpu().numpy())
            predicted_classes = np.argmax(outputs.detach().cpu().numpy(), axis=1)  # get class from network's prediction
            list_pred_train.append(predicted_classes)

            loss.backward() # accumulates the gradient (by addition) for each parameter.
            optimizer.step() # performs a parameter update based on the current gradient

            # getting training quality data
            current_loss = loss.item()
            total_loss += current_loss

        # ----------------- VALIDATION  -----------------

        if epoch > 0 and (epoch % print_every == 0 or epoch == epochs - 1):

            # set model to evaluating (testing)
            model.eval()
            with torch.no_grad():
                val_loss = 0
                list_y, list_pred = [], []
                val_outputs = []
                list_hist = []
                for i, data in enumerate(val_loader):
                    X = data[0].to(device)
                    y = data[1].to(device)

                    outputs, hist = model(X) # this gets the prediction from the network
                    list_hist.append(hist.detach().cpu().numpy())
                    val_outputs.append(outputs.cpu().numpy())
                    predicted_classes = np.argmax(outputs.cpu().numpy(), axis=1) # get class from network's prediction

                    val_loss += loss_function(torch.clip(outputs, 1e-9, 1-1e-9), y).item() # MODIFIED - added [][]

                    if train_dataset.binary_bool:
                        list_y.append(np.argmax(y.cpu().numpy(), axis=1))
                    else:
                        list_y.append(y.cpu().numpy())
                    list_pred.append(predicted_classes)
            scheduler.step(val_loss)

            print(f"Epoch {epoch+1}/{epochs}, training loss: {total_loss/batches:.4f}, validation loss: {val_loss/val_batches:.4f}")
            losses.append(total_loss/batches) # for plotting learning curve
            val_losses.append(val_loss/val_batches) # for plotting learning curve
            
    fig, ax = plt.subplots(1, 1)
    x = np.arange(0, epochs, print_every)
    ax.plot(x, losses, label='train')
    ax.plot(x, val_losses, '--', label='val')
    ax.legend()
    ax.set_ylabel('Loss')
    plt.savefig(os.path.join(MODELFOLDER, f'loss_curves.png'))
    plt.close()

    print(f"Training time: {time.time()-start_ts}s")

    # Save initial network
    # print('Saving initial network')
    # model.save(os.path.join(run_path, "initial_model.h5"))
    #
    # # Initialize training callbacks
    # # history_callback = LossHistory(run_path=run_path)
    # reduce_lr_callback = ReduceLROnPlateau(monitor="val_loss",
    #                                        factor=reduce_lr_factor,
    #                                        patience=reduce_lr_patience,
    #                                        verbose=1, mode="auto",
    #                                        epsilon=reduce_lr_min_delta,
    #                                        cooldown=reduce_lr_cooldown,
    #                                        min_lr=reduce_lr_min_lr)
    # if save_every_epoch:
    #     save_string = "weights/weights.{epoch:03d}-{val_loss:.9f}.h5"
    #     checkpointer = ModelCheckpoint(filepath=os.path.join(run_path,
    #                                    save_string), verbose=1,
    #                                    save_best_only=False)
    # else:
    #     checkpointer = ModelCheckpoint(filepath=os.path.join(run_path,
    #                                    "best_model.h5"), verbose=1,
    #                                    save_best_only=True)
    #
    # # Train!
    # print('Training')
    # t0_train = time()
    # training = model.fit(X, Y, batch_size=batch_size, epochs=epochs,
    #                      verbose=1, validation_data=(val_X, val_Y),
    #                      callbacks=[history_callback, checkpointer,
    #                                 reduce_lr_callback])
    #
    # # Compute total elapsed time for training
    # elapsed_train = time() - t0_train
    # print("Total runtime: %.1f mins" % (elapsed_train / 60))
    #
    # # Save final model
    # print('Saving final model')
    # model.history = history_callback.history
    # model.save(os.path.join(run_path, "final_model.h5"))


if __name__ == '__main__':
    BASEFOLDER = "/home/france/Mounted_dir/results_behavior/MarkerBasedImputation"
    DATASETPATH = '/projects/ag-bozek/france/results_behavior/datasets/INH_FL2_keypoints_1_60_wresiduals_w1nan_stride0.5_new'
    MODELFOLDER = os.path.join(BASEFOLDER, "models")
    PREDICTIONSPATH = os.path.join(BASEFOLDER, "predictions")
    MERGEDFILE = os.path.join(BASEFOLDER, "/fullDay_model_ensemble.h5")

    # Training
    NMODELS = 10
    TRAINSTRIDE = 5
    EPOCHS = 30

    # # Imputation
    # NFOLDS = 20
    # IMPUTESTRIDE = 5
    # ERRORDIFFTHRESH = .5

    # $1 - -base - output - path =$2 - -epochs =$3 - -stride =$4
    # $DATASETPATH $MODELBASEOUTPUTPATH $EPOCHS $TRAINSTRIDE

    train(DATASETNAME, base_output_path=MODELFOLDER, run_name=None,
          data_name=None, net_name="wave_net", clean=False, input_length=9,
          output_length=1, n_markers=60, stride=TRAINSTRIDE, train_fraction=.85,
          val_fraction=0.15, only_moving_frames=False, n_filters=512,
          filter_width=2, layers_per_level=3, n_dilations=None,
          latent_dim=750, epochs=EPOCHS, batch_size=1000,
          lossfunc='mean_squared_error', lr=1e-4, batches_per_epoch=0,
          val_batches_per_epoch=0, reduce_lr_factor=0.5, reduce_lr_patience=3,
          reduce_lr_min_delta=1e-5, reduce_lr_cooldown=0,
          reduce_lr_min_lr=1e-10, save_every_epoch=False, device=torch.device('cuda:0'))