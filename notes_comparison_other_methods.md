# Comparison of DISK with other methods (Keypoint-Moseq, OptiPose, MarkerBasedImputation)
--
Author: France ROSE, February 2025

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

### Testing
