from time import time

t0 = time()
import os, sys
import tqdm
from glob import glob
import argparse
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
    logging.info(f'\n{"-" * 60}'
                 f'\nBASEFOLDER: {BASEFOLDER}\n'
                 f'DATASETPATH: {DATASETPATH}\n'
                 f'front_point = {front_point}\n'
                 f'middle_point = {middle_point}\n'
                 f'train_file = {train_file}\n'
                 f'val_file = {val_file}\n'
                 f'MODELFOLDER = {MODELFOLDER}\n'
                 f'NMODELS = {NMODELS}\n'
                 f'EPOCHS = {EPOCHS}\n'
                 f'training_stride = {TRAINSTRIDE}\n'
                 f'impute_stride = {impute_stride}\n'
                 # f'errordiff_th = {errordiff_th}\n'
                 f'device = {device}\n')

def check_exist(*args):
    print('[check_exist]', args)
    for el in args:
        if type(el) == str:
            if not os.path.exists(el):
                if '.csv' in el:
                    raise ValueError(f'File {el} not found')
                else:
                    os.mkdir(el)
        elif type(el) == list:
            if len(el) == 0:
                raise ValueError(f'Input file list {el} is empty')
            check_exist(*el)
    return


if __name__ == '__main__':

    ##########################################################################################################
    ### CHOOSE DATASET BY SUPPLYING THE COMMANDLINE ARGUMENT
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('dataset', type=str,
                        help='dataset name', choices=['FL2', 'CLB', 'DANNCE', 'Mocap', 'DF3D', 'Fish', 'MABe'])

    parser.add_argument('--train', '-t', action='store_true', default=False,
                        help='retrain')

    args = parser.parse_args()
    ###################################################################################################################
    ## PARAMETERS TO CHANGE FOR THE RUN

    if args.dataset == 'FL2':
        # FL2
        BASEFOLDER = os.path.join(basedir, "results_behavior/MarkerBasedImputation_FL2")
        DATASETPATH = os.path.join(basedir, 'results_behavior/datasets/INH_FL2_keypoints_1_60_wresiduals_w1nan_stride0.5_new')
        front_point = ['left_coord', 'right_coord']
        middle_point = ['left_hip', 'right_hip']
        TRAINSTRIDE = 1 #5 # FL2 is a smaller dataset than they had (25 million frames for training)

        short_seq_datafile = os.path.join(basedir, 'results_behavior/outputs/25-09-24_FL2_new_for_comparison/DISK_test/test_for_optipose_repeat_0/test_repeat-0.csv')
        long_seq_datafiles = glob(os.path.join(basedir, 'results_behavior/outputs/25-09-24_FL2_new_for_comparison/DISK_test/test_for_optipose_repeat_0/test_w-all-nans_file*.csv'))

    elif args.dataset == 'DANNCE':
        # DANNCE
        BASEFOLDER = os.path.join(basedir, "results_behavior/MarkerBasedImputation_DANNCE/")
        DATASETPATH = os.path.join(basedir, 'results_behavior/datasets/DANNCE_seq_keypoints_60_stride30_fill10_new')
        front_point = 'SpineF'
        middle_point = 'SpineM'
        TRAINSTRIDE = 5 # FL2 is a smaller dataset than they had (25 million frames for training)

        short_seq_datafile = os.path.join(basedir, 'results_behavior/outputs/13-02-25_DANNCE_for_comparison/DISK_test/test_for_optipose_repeat_0/test_repeat-0.csv')
        long_seq_datafiles = glob(os.path.join(basedir,
                                               'results_behavior/outputs/13-02-25_DANNCE_for_comparison/DISK_test/test_for_optipose_repeat_0/test_fulllength_dataset_w-all-nans_file-*.csv'))

    elif args.dataset == 'CLB':
        # CLB
        BASEFOLDER = os.path.join(basedir, "results_behavior/MarkerBasedImputation_CLB/")
        DATASETPATH = os.path.join(basedir, 'results_behavior/datasets/INH_CLB_keypoints_1_60_stride0.5')
        front_point = ['left_coord', 'right_coord']
        middle_point = ['left_hip', 'right_hip']
        TRAINSTRIDE = 1  # FL2 is a smaller dataset than they had (25 million frames for training)

        short_seq_datafile = os.path.join(basedir,
                                          'results_behavior/outputs/13-02-25_CLB_for_comparison/DISK_test/test_for_optipose_repeat_0/test_repeat-0.csv')
        long_seq_datafiles = glob(os.path.join(basedir,
                                               'results_behavior/outputs/13-02-25_CLB_for_comparison/DISK_test/test_for_optipose_repeat_0/test_fulllength_dataset_w-all-nans_file-*.csv'))

    elif args.dataset == 'MABe':
        # MABe
        BASEFOLDER = os.path.join(basedir, "results_behavior/MarkerBasedImputation_MABe/")
        DATASETPATH = os.path.join(basedir, 'results_behavior/datasets/MABE_task1_60stride60')
        front_point = ['kp3_animal0', 'kp3_animal1']
        middle_point = ['kp6_animal0', 'kp6_animal1']
        TRAINSTRIDE = 1  # FL2 is a smaller dataset than they had (25 million frames for training)

        short_seq_datafile = os.path.join(basedir,
                                          'results_behavior/outputs/2024-02-19_MABe_task1_newnewmissing/DISK_test/test_for_optipose_repeat_0/test_repeat-0.csv')
        long_seq_datafiles = glob(os.path.join(basedir,
                                               'results_behavior/outputs/2024-02-19_MABe_task1_newnewmissing/DISK_test/test_for_optipose_repeat_0/test_fulllength_dataset_w-all-nans_file-*.csv'))

    elif args.dataset == 'DF3D':
        # DF3D
        BASEFOLDER = os.path.join(basedir, "results_behavior/MarkerBasedImputation_DF3D/")
        DATASETPATH = os.path.join(basedir, 'results_behavior/datasets/DF3D_keypoints_60stride5_new')
        front_point = ['15', '34'] # cf image on https://github.com/NeLy-EPFL/DeepFly3D
        middle_point = ['16', '35']
        TRAINSTRIDE = 1  # FL2 is a smaller dataset than they had (25 million frames for training)

        short_seq_datafile = os.path.join(basedir,
                                          'results_behavior/outputs/2025-02-13_DF3D_for_comparison/DISK_test/test_for_optipose_repeat_0/test_repeat-0.csv')
        long_seq_datafiles = glob(os.path.join(basedir,
                                               'results_behavior/outputs/2025-02-13_DF3D_for_comparison/DISK_test/test_for_optipose_repeat_0/test_fulllength_dataset_w-all-nans_file-*.csv'))
    elif args.dataset == 'Fish':
        # Fish
        BASEFOLDER = os.path.join(basedir, "results_behavior/MarkerBasedImputation_Fish/")
        DATASETPATH = os.path.join(basedir, 'results_behavior/datasets/Fish_v3_60stride120')
        front_point = ['fish1_head', 'fish2_head']
        middle_point = ['fish1_tail', 'fish2_tail']
        TRAINSTRIDE = 50  # FL2 is a smaller dataset than they had (25 million frames for training)

        short_seq_datafile = os.path.join(basedir,
                                          'results_behavior/outputs/2023-09-27_Fishv3_newnewmissing/DISK_test_for_comparison/test_for_optipose_repeat_0/test_repeat-0.csv')
        long_seq_datafiles = glob(os.path.join(basedir,
                                               'results_behavior/outputs/2023-09-27_Fishv3_newnewmissing/DISK_test_for_comparison/test_for_optipose_repeat_0/test_fulllength_dataset_w-all-nans_file-*.csv'))

    elif args.dataset == 'Mocap':
        ## Mocap
        BASEFOLDER = os.path.join(basedir, "results_behavior/MarkerBasedImputation_Mocap/")
        DATASETPATH = os.path.join(basedir, 'results_behavior/datasets/Mocap_keypoints_60_stride30_new')
        front_point = 'arm1_1' # names got shuffled, spine0
        middle_point = 'arm1_0'# spine1
        TRAINSTRIDE = 1

        short_seq_datafile = os.path.join(basedir,
                                          'results_behavior/outputs/2025-02-24_Mocap_for_comparison/DISK_test/test_for_optipose_repeat_0/test_repeat-0.csv')
        long_seq_datafiles = glob(os.path.join(basedir,
                                               'results_behavior/outputs/2025-02-24_Mocap_for_comparison/DISK_test/test_for_optipose_repeat_0/test_fulllength_dataset_w-all-nans_file-*.csv'))

    else:
        sys.exit(1)

    ###################################################################################################################

    check_exist(BASEFOLDER, DATASETPATH, short_seq_datafile, long_seq_datafiles)

    train_file = os.path.join(DATASETPATH, 'train_dataset_w-0-nans.npz')
    val_file = os.path.join(DATASETPATH, 'val_dataset_w-0-nans.npz')
    MODELFOLDER = os.path.join(BASEFOLDER, "models")

    # Training
    NMODELS = 10
    EPOCHS = 30

    # Imputation
    impute_stride = 1 #5
    # errordiff_th = 0.5

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
    models = glob(os.path.join(BASEFOLDER, f'models-wave_net_epochs={EPOCHS}_input_9_output_1*/best_model.h5'))

    # only train the number of missing models
    for _ in range(NMODELS - len(models)):
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

    save_path = os.path.join(BASEFOLDER, 'model_ensemble')
    model_ensemble_path = os.path.join(save_path, 'final_model.h5')
    if not os.path.exists(model_ensemble_path):
        save_path = build_ensemble(BASEFOLDER, models, run_name=None, clean=False, device=device)
        model_ensemble_path = os.path.join(save_path, 'final_model.h5')
    logging.info(f'SAVEPATH = {save_path}')

    # EVALUATION

    # ON DATA LIKE SEEN IN TRAINING
    # test_file_like_training = os.path.join(DATASETPATH, 'test_dataset_w-0-nans.npz')
    # logging.info(f'datafile = {test_file_like_training}')
    # testing_single_model_like_training(test_file_like_training, front_point=front_point, middle_point=middle_point,
    #                 model_name=models[0],
    #                 net_name="wave_net", clean=False, input_length=9,
    #                 output_length=1, stride=1,
    #                 batch_size=1000,
    #                 lossfunc='mean_squared_error',
    #                 device=torch.device('cpu'))
    # testing_single_model_like_predict(models[0], test_file_like_training, DATASETPATH,
    #         save_path = os.path.dirname(models[0]), front_point=front_point, middle_point=middle_point,
    #         model = None, device = torch.device('cpu'))
    # testing_ensemble_model_like_predict(model_ensemble_path, test_file_like_training, DATASETPATH,
    #         save_path = save_path, front_point=front_point, middle_point=middle_point,
    #         model = None, device = torch.device('cpu'))

    # ON SHORT SEQUENCES WITH GROUND TRUTH

    logging.info(f'datafile = {short_seq_datafile}')

    for pass_direction in ['reverse', 'forward']:
        predict_single_pass(model_ensemble_path, short_seq_datafile, DATASETPATH, pass_direction,
                            save_path=save_path, stride=impute_stride,
                            model=None, front_point=front_point, middle_point=middle_point)

    t_after_predict = time()
    logging.info(f'Time predict: {t_after_predict - t_after_training}')

    pred_paths = glob(os.path.join(save_path, 'test_repeat-0*.mat'))
    save_path = os.path.join(save_path, f'{os.path.basename(short_seq_datafile).split(".")[0]}_merged')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    merge(save_path, pred_paths, DATASETPATH)

    t_after_merge = time()
    logging.info(f'Time predict: {t_after_merge - t_after_predict}')


    # ON ORIGINAL FILES FOR REAL-SCENARIO IMPUTATION
    for data_file in long_seq_datafiles:
        if 'model_10_5_1' in data_file or 'model_15_5_1' in data_file:
            continue
        logging.info(f'datafile = {data_file}')

        save_path_tmp = os.path.join(save_path, f'{os.path.basename(data_file).split(".")[0]}_merged')
        if not os.path.exists(save_path_tmp):
            os.mkdir(save_path_tmp)

        for pass_direction in ['reverse', 'forward']:
            predict_single_pass(model_ensemble_path, data_file, DATASETPATH, pass_direction,
                                front_point=front_point, middle_point=middle_point,
                                save_path=save_path_tmp, stride=impute_stride,
                                model=None)

        pred_paths = glob(os.path.join(save_path_tmp, f'{os.path.basename(data_file).split(".csv")[0]}*.mat'))
        merge(save_path_tmp, pred_paths, DATASETPATH)