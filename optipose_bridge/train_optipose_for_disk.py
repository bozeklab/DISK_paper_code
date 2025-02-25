# Pre-generated datasets are provided in the augmented_train_test folder on the Google Drive.
# Usage python model_train_df3d.py n_cm k n_heads

import csv
csv.field_size_limit(1_000_000_000)
import json
import pickle
import os
import numpy as np
import tensorflow as tf
from cvkit.pose_estimation.metrics.mpjpe import build_mpjpe_metric
from cvkit.pose_estimation.metrics.pck import build_pck_metric

from cvkit.pose_estimation.config import PoseEstimationConfig
from OptiPose.model.postural_autoencoder import optipose_postural_autoencoder
from OptiPose.model.utils import train, build_spatio_temporal_loss

import sys

#######################################################################################################################
## Awaits 5 arguments:
print('ARGS', sys.argv)

dataset_title = sys.argv[1] # dataset name to find the config file, should match f'./example_configs/{dataset_title}.yml'
# 3 hyperparameters of the networks (see paper for more details)
numPCMs       = int(sys.argv[2])
num_sub_ck    = int(sys.argv[3])
multi_heads   = int(sys.argv[4])
# learning rate
lr            = float(sys.argv[5])
#######################################################################################################################

# base output directory
base_output_dir = '/projects/ag-bozek/france/results_behavior/optipose/'

# load config file
config = PoseEstimationConfig(f'./example_configs/{dataset_title}.yml')
n_kps = config.num_parts # number of keypoints, loaded from the config

# fixed parameters
num_samples = 20000 # number of training samples
batch_size = 100
max_epochs = 1500
length_seq = 60 # length of input sequences

# initialize the model
model = optipose_postural_autoencoder(length_seq, n_kps, numPCMs, num_sub_ck, multi_heads)#, output_dim=120)
# model.summary()

# initialize the metrics
metrics = ["mae", build_pck_metric(n_kps, 0.05), build_pck_metric(n_kps, 0.1), build_mpjpe_metric()]

# construct the datasets from the data files
dataset = {'input': [], 'label': []}
vdataset = {'input': [], 'label': []}
reader = csv.reader(open(config.train_file), delimiter='|')
reader1 = csv.reader(open(config.val_file), delimiter='|')

next(reader)
next(reader1)

for row in reader:
    dataset["input"].append(np.array(json.loads(row[0])))
    dataset["label"].append(np.array(json.loads(row[1])))

for row in reader1:
    vdataset["input"].append(np.array(json.loads(row[0])))
    vdataset["label"].append(np.array(json.loads(row[1])))

xTrain = dataset['input']
yTrain = dataset['label']
xVal = vdataset['input']
yVal = vdataset['label']

train_dataset = tf.data.Dataset.from_tensor_slices((xTrain, yTrain)).batch(batch_size).shuffle(15000,
                                                                                               reshuffle_each_iteration=True)
validation_dataset = tf.data.Dataset.from_tensor_slices((xVal, yVal)).batch(batch_size)

# learning scheduler
def scheduler(epoch, lr):
    if epoch % 40 == 0 and epoch < 1000:
        return lr * 0.95
    elif epoch % 300 == 0 and epoch < 3000:
        return lr * 0.95
    elif epoch % 300 == 0:
        return lr * 0.95
    else:
        return lr

callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

# configurate the saving parameters
output_dir = os.path.join(base_output_dir, f"{dataset_title}/model_weights/optipose-{numPCMs}-{num_sub_ck}-{dataset_title}-{multi_heads}_{lr}")
checkpoint_path = output_dir + "/closs-{epoch:04d}.ckpt"
callback_1 = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=0,
    save_weights_only=True,
    save_best_only=True,
    mode='max',
    monitor='val_pck_5')
termination = tf.keras.callbacks.TerminateOnNaN()

# train
history = train(model, train_dataset, validation_dataset, build_spatio_temporal_loss(0.0001, 0.0001), max_epochs, lr,
                metrics, [callback, callback_1, termination])

# save training history
with open(output_dir + f'/optipose-{numPCMs}-{num_sub_ck}-{dataset_title}-{multi_heads}-history', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
