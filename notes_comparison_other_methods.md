# Comparison of DISK with other methods (Keypoint-Moseq, OptiPose, MarkerBasedImputation)
--
Author: France ROSE, February 2025

## Overall strategy

- For each dataset the different methods (Keypoint-Moseq, OptiPose, MarkerBasedImputation, DISK) are trained on the same data (training data without keypoints missing).
- For MarkerBasedImputation, I had to reimplement the code as I could not run the codebase with old keras dependencies (see below).
- For Keypoint-Moseq and Optipose, I used the original implementation.

### Testing

To compare the performance of MarkerBasedImputation and DISK. 
I generated short sequences (60 frames) and random gaps with the DISK testing script and save them in a csv file (DISK repo branch optipose).
These sequences are then preprocessed for the MBI approach (centering, rotation, and z-scoring). 
To avoid leakage between the input sequences and the corresponding ground truth, the preprocessing is computed on the masked segments directly.

The metrics (PCK, RMSE, MPJPE) are computed on original coordinates, as each method can have a different preprocessing and normalization technique.

### Inference

I also ran the methods on the real gaps in the data. Here metrics cannot be computed but imputed trajectories can be visualized.
The preprocessing was also done according to the chosen method.


## MarkerBasedImputation

- I based my re-implementation on the repo and the CAPTURE paper
- I could not pip install and use out of the box the repo as the versions of keras and tensorflow were not compatible with our high computer cluster
- I re-implemented part of it in pytorch (see folder `MarkerBasedImputation_bridge`)

### Processing of data

- The processing of data is done upstream from the MarkerBasedImputation repo (imports directly "aligned_markers" in script `genDataset`)
- The processing has two steps
  1. median filtering
  2. centering and rotating 
  3. z-scoring

The **step 2.** is happening for each time step independently. 

The centering is done with respect to a defined point middle of the skeleton, for the original skeleton it was `spineF`. 
For the FL2 and CLB dataset, we defined it as the center point between the left hip and the righ hip marker. 
As the central point will be different for each dataset/skeleton, I made it a parameter the user of the script has to set.

The rotation is made with respect to the vector between a "middle" point and a "front" point. In the same fashion, the front point has to be set by the user as it is dataset-specific.

The **step 3.** is done per recording in the original version. 
With this implementation choice, I ran into problems when testing short sequences of 60 frames as it is the case for DISK.
I decided to compute the z-score parameters on the entire training / test set with potential missing data, so the transformation is more robust to short sequences and missing data. 
I understood the goal of this step as a way to normalize the input values to the neural network, so the exact implementation choice should not matter too much.

### Reimplementation of the model and training

I simply ported the model form keras to pytorch.
I changed the convolution layers to residual convolution layers as it boosted the performance by a lot.
I kept the other (hyper)parameters similar to the original ones.

To build the ensemble model, I also copied the original code.



## Optipose

### Conda environment

cf `OPTIPOSE_conda_env_2025-02.yml` - uses tensorflow and is different from DISK conda environment.
You can use the yml file to install the environment using `conda env create --file OPTIPOSE_conda_env_2025-02.yml`
I could not make it work on our setup with GPU, making the training quite slow.

For DISK environment installation and usage see `github.com/DISK`


### Prepare the files

The `optipose_bridge/create_csv_for_optipose.py` script converts the npz DISK dataset files (with all the holes) into csv files. 
I only select holes that are smaller than the considered segment length of 60 frames (for testing purposes).

T

### Config

The config file (in folder `example_configs`) will be used not only for the training but also reconstruction.

### Train

```
conda activate OPTIPOSE
cd ~/DISK_paper_code/optipose_bridge/
python ~/DISK_paper_code/optipose_bridge/train_optipose_for_disk.py DANNCE 15 5 4 3.1e-4
```

When setting the learning rate "too high", I experienced an explosion of the loss leading to NaNs.
I lowered consequently the learning rate to avoid this behavior (to 1e-4)

### Run inference


## Keypoint-Moseq


