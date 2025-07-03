# FIG3_FILES fOLDER EXPLANATION

Here are some files to support / reproduce the figure 3 in the DISK paper.

## Model

- The model used to make the plots is the one saved in the `model_epoch1370` checkpoint and was trained under the `config_for_train.yaml` configuration.

## Evaluation and plots

- Then this model was evaluated on the test set using `config_for_test.yaml` configuration.
- the .png snv .svg files are output files of the test script that were used in the final figures.
- the exact samples displayed in `reconstruction_xyz_...` cannot be exactly reproduced as the gaps are created on the fly while using the test script. However similar results can be obtained.
- the `total_RMSE_repeat-0.csv` records all the gaps and prediction errors that were generated. With this file, custom plots reproducing the error per keypoint, or the error wrt to the gap length can easily be produces.
- the `mean_RMSE.csv` only contains the mean RMSE error per model.