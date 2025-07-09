import numpy as np
import argparse
import os, sys
from glob import glob
from omegaconf import OmegaConf
import logging

from sklearn.metrics import confusion_matrix, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

from DISK.embedding_umap import extract_hidden
from DISK.utils.utils import read_constant_file, load_checkpoint
from DISK.utils.dataset_utils import load_datasets
from DISK.utils.transforms import init_transforms
from DISK.utils.train_fillmissing import construct_NN_model
from DISK.models.graph import Graph

import torch
from torch.utils.data import DataLoader


if __name__ == '__main__':

    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--checkpoint_folder", type=str, required=True)
    p.add_argument("--stride", type=float, required=True, default='in seconds')
    p.add_argument("--n_seeds", type=int, default=1, desc='number of random forests run')

    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format=f'[%(levelname)s][%(asctime)s] %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("numpy").setLevel(logging.WARNING)

    basedir = '/projects/ag-bozek/france/results_behavior'
    if not os.path.exists(basedir):
        basedir = '/home/france/Mounted_dir/results_behavior'

    config_file = os.path.join(args.checkpoint_folder, '.hydra', 'config.yaml')
    model_cfg = OmegaConf.load(config_file)
    model_path = glob(os.path.join(args.checkpoint_folder, 'model_epoch*'))[0]  # model_epoch to not take the model from the lastepoch

    dataset_constants = read_constant_file(os.path.join(basedir, 'datasets', model_cfg.dataset.name, 'constants.py'))

    if model_cfg.dataset.skeleton_file is not None:
        skeleton_file_path = os.path.join(basedir, 'datasets', model_cfg.dataset.skeleton_file)
        if not os.path.exists(skeleton_file_path):
            raise ValueError(f'no skeleton file found in', skeleton_file_path)
        skeleton_graph = Graph(file=skeleton_file_path)
    else:
        skeleton_graph = None
        skeleton_file_path = None

    """ DATA """
    transforms, _ = init_transforms(model_cfg, dataset_constants.KEYPOINTS, dataset_constants.DIVIDER,
                                 dataset_constants.SEQ_LENGTH, basedir, args.checkpoint_folder)



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Device: {}".format(device))


    logging.info('Loading transformer model...')
    # load model
    model = construct_NN_model(model_cfg, dataset_constants, skeleton_file_path, device)
    logging.info(f'Network constructed')

    load_checkpoint(model, None, model_path, device)


    logging.info('Loading datasets...')
    train_dataset, val_dataset, test_dataset = load_datasets(dataset_name=model_cfg.dataset.name,
                                                             dataset_constants=dataset_constants,
                                                             transform=transforms,
                                                             dataset_type='full_length',
                                                             stride=args.stride,
                                                             suffix='_w-0-nans',
                                                             root_path=basedir,
                                                             length_sample=dataset_constants.SEQ_LENGTH,
                                                             freq=dataset_constants.FREQ,
                                                             outputdir=args.checkpoint_folder,
                                                             skeleton_file=None,
                                                             label_type='all',  # don't care, not using
                                                             verbose=model_cfg.feed_data.verbose)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    logging.info('Extract hidden representation...')
    ### DIRECT KNN ON SEQ2SEQ LATENT SPACE
    hi_train, label_train, index_file_train, index_pos_train, statistics_train = extract_hidden(model, train_loader,
                                                                                                dataset_constants,
                                                                                                model_cfg,
                                                                                                device,
                                                                                                compute_statistics=True)
    logging.info('Done with train hidden representation...')

    time_train = train_dataset.possible_times
    i_file_train = train_dataset.possible_indices[:, 0]
    hi_eval, label_eval, index_file_eval, index_pos_eval, statistics_eval = extract_hidden(model,
                                                                                           val_loader,
                                                                                           dataset_constants,
                                                                                           model_cfg,
                                                                                           device,
                                                                                           compute_statistics=True)
    logging.info('Done with val hidden representation...')

    time_eval = val_dataset.possible_times
    i_file_eval = val_dataset.possible_indices[:, 0]

    logging.info(f'hidden vectors eval {hi_eval.shape} {label_train.shape} {np.unique(label_train, return_counts=True)}')
    logging.info(f'hidden train eval {hi_train.shape} {label_eval.shape} {np.unique(label_eval, return_counts=True)}')


    ################################################################

    ## for Human MOCAP
    # reverse_dict_label = {0: 'Walk', 1: 'Wash', 2: 'Run', 3: 'Jump', 4: 'Animal Behavior', 5: 'Dance',
    #                       6: 'Step', 7: 'Climb', 8: 'unknown'}
    # y_train = label_train[:, -1]
    # y_eval = label_eval[:, -1]
    # logging.info(f'unique y values {np.unique(y_train)} {np.unique((y_eval))}')
    # X_train = hi_train[y_train != 8, :]
    # y_train = y_train[y_train != 8]
    # X_eval = hi_eval[y_eval != 8, :]
    # y_eval = y_eval[y_eval != 8]

    ## for MABe
    y_train = label_train[:, 0]
    y_eval = label_eval[:, 0]
    logging.info(f'unique y values {np.unique(y_train, return_counts=True)} {np.unique(y_eval, return_counts=True)}')
    X_train = hi_train
    X_eval = hi_eval

    ################################################################
    logging.info(f'final shapes {y_train.shape} {y_eval.shape}')

    bal_acc = []
    f1 = []
    precision = []
    recall = []
    conf_mat = None
    for seed in np.arange(0, args.n_seeds, dtype=int):
        rfc = RandomForestClassifier(random_state=seed)

        rfc.fit(X_train, y_train)
        y_eval_predict = rfc.predict(X_eval)

        bal_acc.append(balanced_accuracy_score(y_eval_predict, y_eval))
        f1.append(f1_score(y_eval_predict, y_eval, average="macro"))
        precision.append(precision_score(y_eval_predict, y_eval, average="macro"))
        recall.append(recall_score(y_eval_predict, y_eval, average="macro"))
        if seed == 0:
            conf_mat = confusion_matrix(y_eval, y_eval_predict)
        else:
            conf_mat += confusion_matrix(y_eval, y_eval_predict)
        logging.info(f'RandomForest results SEED {seed}:\n'
                     f'Balanced Accuracy: {bal_acc[-1]:.3f}\n'
                     f'Balanced F1 score: {f1[-1]:.3f}\n'
                     f'Balanced Precision score: {precision[-1]:.3f}\n'
                     f'Balanced Recall score: {recall[-1]:.3f}\n'
                     f'Confusion matrix: {conf_mat}\n')
    logging.info(f'RandomForest results:\n'
                 f'Balanced Accuracy: {np.mean(bal_acc):.3f} +/- {np.std(bal_acc):.3f}\n'
                 f'Balanced F1 score: {np.mean(f1):.3f} +/- {np.std(f1):.3f}\n'
                 f'Balanced Precision score: {np.mean(precision):.3f} +/- {np.std(precision):.3f}\n'
                 f'Balanced Recall score: {np.mean(recall):.3f} +/- {np.std(recall):.3f}\n'
                 f'Confusion matrix: {conf_mat / n_seeds}\n')


