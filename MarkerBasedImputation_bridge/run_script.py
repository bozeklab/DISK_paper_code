from time import time

t0 = time()
import os, sys
import tqdm
from glob import glob

import numpy as np
import pandas as pd
from skimage.io import imread, imsave

import matplotlib.pyplot as plt
import matplotlib
if os.uname().nodename == 'france-XPS':
    matplotlib.use('TkAgg')
    basedir = '/home/france/Mounted_dir'
else:
    matplotlib.use('Agg')
    basedir = '/projects/ag-bozek/france'

import torch
from training import train
from testing import testing_single_model_like_predict, testing_ensemble_model_like_predict
from testing2 import testing_single_model_like_training
from build_ensemble import build_ensemble
from predict_single_pass import predict_single_pass
from merge import merge
import logging
t_after_import = time()

def write_logging():
    logging.info(f'BASEFOLDER: {BASEFOLDER}\n'
                 f'DATASETPATH: {DATASETPATH}\n'
                 f'front_point = {front_point}\n'
                 f'middle_point = {middle_point}\n'
                 f'train_file = {train_file}\n'
                 f'val_file = {val_file}\n'
                 f'MODELFOLDER = {MODELFOLDER}\n'
                 f'NMODELS = {NMODELS}\n'
                 f'TRAINSTRIDE = {TRAINSTRIDE}\n'
                 f'EPOCHS = {EPOCHS}\n'
                 f'impute_stride = {impute_stride}\n'
                 f'errordiff_th = {errordiff_th}\n'
                 f'device = {device}\n')


if __name__ == '__main__':
    BASEFOLDER = os.path.join(basedir, "results_behavior/MarkerBasedImputation_FL2")
    # BASEFOLDER = os.path.join('/home/france/Documents', "MarkerBasedImputation_FL2")
    if not os.path.exists(BASEFOLDER):
        os.mkdir(BASEFOLDER)
    # DATASETPATH = os.path.join(basedir, 'results_behavior/datasets/DANNCE_seq_keypoints_60_stride30_fill10')
    DATASETPATH = os.path.join(basedir, 'results_behavior/datasets/INH_FL2_keypoints_1_60_wresiduals_w1nan_stride0.5_new')
    # DATASETPATH = os.path.join('/home/france/Documents', 'INH_FL2_keypoints_1_60_wresiduals_w1nan_stride0.5_new')
    front_point = ['left_coord', 'right_coord'] #'SpineF'
    middle_point = ['left_hip', 'right_hip'] #'SpineM'
    train_file = os.path.join(DATASETPATH, 'train_dataset_w-0-nans.npz')
    val_file = os.path.join(DATASETPATH, 'val_dataset_w-0-nans.npz')
    MODELFOLDER = os.path.join(BASEFOLDER, "models")

    # Training
    NMODELS = 10
    TRAINSTRIDE = 1 #5 # FL2 is a smaller dataset than they had (25 million frames for training)
    EPOCHS = 30 #30

    # Imputation
    impute_stride = 1 #5
    errordiff_th = 0.5

    print(f'Time imports: {t_after_import - t0}')
    device = torch.device('cuda:0')

    logging.basicConfig(level=logging.INFO,
                        format=f'[%(levelname)s][%(asctime)s] %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S',
                        filename=os.path.join(BASEFOLDER, 'run_script.log'))
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("numpy").setLevel(logging.WARNING)

    write_logging()

    # TRAINING
    for _ in range(NMODELS):
        train(train_file, val_file, front_point=front_point, middle_point=middle_point,
              base_output_path=MODELFOLDER, run_name=None,
              data_name=None, net_name="wave_net", clean=False, input_length=9,
              output_length=1, stride=TRAINSTRIDE, train_fraction=.85,
              val_fraction=0.15, only_moving_frames=False, n_filters=512,
              filter_width=2, layers_per_level=3, n_dilations=None,
              latent_dim=750, epochs=EPOCHS, batch_size=1000,
              lossfunc='mean_squared_error', lr=1e-4, batches_per_epoch=0,
              val_batches_per_epoch=0, reduce_lr_factor=0.5, reduce_lr_patience=3,
              reduce_lr_min_delta=1e-5, reduce_lr_cooldown=0,
              reduce_lr_min_lr=1e-10, save_every_epoch=False, device=device)

    t_after_training = time()
    logging.info(f'Time training: {t_after_training - t_after_import}')

    models = glob(os.path.join(BASEFOLDER, f'models-wave_net_epochs={EPOCHS}_input_9_output_1*/best_model.h5'))
    if len(models) == 0:
        print(f"no models found at {os.path.join(BASEFOLDER, f'models-wave_net_epochs={EPOCHS}_input_9_output_1*/best_model.h5')}")
        logging.info(f"no models found at {os.path.join(BASEFOLDER, f'models-wave_net_epochs={EPOCHS}_input_9_output_1*/best_model.h5')}")
        sys.exit(1)

    save_path = build_ensemble(BASEFOLDER, models, run_name=None, clean=False, device=device)
    # save_path = os.path.join(BASEFOLDER, 'model_ensemble_10')
    model_ensemble_path = os.path.join(save_path, 'final_model.h5')
    logging.info(f'SAVEPATH = {save_path}')

    # EVALUATION

    # ON DATA LIKE SEEN IN TRAINING
    # data_file = os.path.join(DATASETPATH, 'val_dataset_w-0-nans.npz')
    data_file = os.path.join(DATASETPATH, 'test_dataset_w-0-nans.npz')
    logging.info(f'datafile = {data_file}')
    testing_single_model_like_training(data_file, front_point=front_point, middle_point=middle_point,
                    model_name=models[0],
                    net_name="wave_net", clean=False, input_length=9,
                    output_length=1, stride=1,
                    batch_size=1000,
                    lossfunc='mean_squared_error',
                    device=torch.device('cpu'))
    testing_single_model_like_predict(models[0], data_file, DATASETPATH,
            save_path = os.path.dirname(models[0]), stride = 1, n_folds = 1, fold_id = 0,
            markers_to_fix = None, error_diff_thresh = errordiff_th,
            model = None, device = torch.device('cpu'))
    testing_ensemble_model_like_predict(model_ensemble_path, data_file, DATASETPATH,
            save_path = save_path, stride = 1, n_folds = 1, fold_id = 0,
            markers_to_fix = None, error_diff_thresh = errordiff_th,
            model = None, device = torch.device('cpu'))

    # ON SHORT SEQUENCES WITH GROUND TRUTH
    data_file = os.path.join(basedir, 'results_behavior/outputs/25-09-24_FL2_new_for_comparison/DISK_test/test_for_optipose_repeat_0/test_repeat-0.csv')
    logging.info(f'datafile = {data_file}')

    for pass_direction in ['forward', 'reverse']:
        predict_single_pass(model_ensemble_path, data_file, DATASETPATH, pass_direction,
                            save_path=save_path, stride=impute_stride, n_folds=1, fold_id=0,
                            markers_to_fix=None, error_diff_thresh=errordiff_th,
                            model=None, front_point=front_point, middle_point=middle_point)

    t_after_predict = time()
    logging.info(f'Time predict: {t_after_predict - t_after_training}')

    fold_paths = glob(os.path.join(save_path, 'test_repeat-0*.mat'))
    save_path = os.path.join(save_path, f'{os.path.basename(data_file).split(".")[0]}_merged')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    merge(save_path, fold_paths)

    t_after_merge = time()
    logging.info(f'Time predict: {t_after_merge - t_after_predict}')


    # ON ORIGINAL FILES FOR REAL-SCENARIO IMPUTATION
    for data_file in glob(os.path.join(basedir, 'results_behavior/outputs/25-09-24_FL2_new_for_comparison/DISK_test/test_for_optipose_repeat_0/test_w-all-nans_file*.csv')):
        if 'model_10_5_1' in data_file:
            continue
        logging.info(f'datafile = {data_file}')

        save_path_tmp = os.path.join(save_path, f'{os.path.basename(data_file).split(".")[0]}_merged')
        if not os.path.exists(save_path_tmp):
            os.mkdir(save_path_tmp)

        for pass_direction in ['reverse', 'forward']:
            predict_single_pass(model_ensemble_path, data_file, DATASETPATH, pass_direction,
                                save_path=save_path_tmp, stride=impute_stride, n_folds=1, fold_id=0,
                                markers_to_fix=None, error_diff_thresh=errordiff_th,
                                model=None)

        fold_paths = glob(os.path.join(save_path_tmp, f'{os.path.basename(data_file).split(".csv")[0]}*.mat'))
        merge(save_path_tmp, fold_paths)