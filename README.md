# DISK paper code
Additional scripts to reproduce analyses from the DISK paper.
DISK code is available at: https://github.com/bozeklab/DISK

## List of figures and related code

| Figure                  | Title                                                                                                                  | Script files                                   | Comments                                                                                                |
|-------------------------|------------------------------------------------------------------------------------------------------------------------|------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| Fig. 1                  | The missing data problem                                                                                               | -                                              | Measure % missing per dataset                                                                           |
| Fig. 2 a & f            | DISK and other imputation methods’ performance across datasets                                                         | `fig2_barplots.py`                             | outputs of DISK main code (`test_fillmissing` script); input files in `fig2a_result_files` folder       |
| Fig. 2 b - e & g        | DISK and other imputation methods’ performance across datasets                                                         | `final_comparison_methods_short_segment.py`    | input files in `comparison_methods_files`                                                               |
| Fig. 2 h & i            | DISK and other imputation methods’ performance across datasets                                                         | `final_comparison_methods_full_file.py`        | input files in `comparison_methods_files`                                                               |
| Fig. 3                  | Estimated imputation error                                                                                             | -                                              | outputs of DISK main code (`test_fillmissing` script); input files in `fig3_files` folder               |
| Fig. 4 a                | Inference of multiple simultaneously missing keypoints                                                                 | `fig_4a_nmissing.py`                           | input files in `fig4_files` folder                                                                      | 
| Fig. 4 c                | Inference of multiple simultaneously missing keypoints                                                                 | `fig_4b.py`                                    | input files in `fig4_files` folder                                                                      | 
| Fig. 4 d-f              | Inference of multiple simultaneously missing keypoints                                                                 | -                                              | outputs of DISK main code (`test_fillmissing` script)                                                   |
| Fig. 5                  | DISK learns meaningful representations of sequences from  Human and 2-Mice-2D datasets                                 | -                                              | outputs of DISK main code (`embedding_umap` script); checkpoints and other files in `fig5_files` folder |
| Fig. 6                  | DISK allows to detect more steps and emphasizes differences in step kinematics between different treatments            | folder `fig6_step_detection`                   |                                                                                                         |
| Ext. Fig. 1 a - b       | Comparison of different architectures [...] and of different methods according to the MPJPE and PCK@0.01 metrics.      | `fig2_barplots.py`                             |                                                                                                         |
| Ext. Fig. 1 c - d       | Comparison of different architectures [...] and of different methods according to the MPJPE and PCK@0.01 metrics.      | `final_comparison_methods_short_segment.py`    |                                                                                                         |
| Ext. Fig. 2             | PCK@0.01, RMSE, and MPJPE per keypoint and averaged across keypoints (”all”) in datasets                               | `fig2_barplots.py`                             |                                                                                                         |
| Ext. Fig. 3 (upper row) | Test RMSE with respect to the gap length for the other datasets.                                                       | -                                              | outputs of DISK main code (`test_fillmissing` script)                                                   |
| Ext. Fig. 3 (lower row) | Test RMSE with respect to the gap length for the other datasets.                                                       | `final_comparison_methods_short_segment.py`    |                                                                                                         |
| Ext. Fig. 4             | DISK performance dependence on dataset size and input length (2-Fish dataset)                                          | `DISK_peformance_dataset_size_input_length.py` |                                                                                                         | 
| Ext. Fig. 5 - 6         | Imputation of switched keypoints.                                                                                      | `analyse_swap_keypoints.py`                    |                                                                                                         | 
| Ext. Fig. 7             | Observed missing proportions for each keypoint in mouse FL2 and CLB datasets                                           | -                                              | outputs of DISK main code (`create_proba_missing` script)                                               | 
| Supp Data               | Analysis of DISK performance with respect to action classes, quantity of movement, periodicity, and gap length         | `DISK_perforamcne_vs_action.py`                |                                                                                                         | 
| Supp Data               | Estimated error correlation plots [...] for DISK-proba and GRU-proba for all tested datasets.                          | -                                              | outputs of DISK main code (`test_fillmissing` script)                                                   | 
| Supp Data               | RMSE wrt the distance between the two fish and the number and scheme of missing keypoints for short gaps and long gaps | `fig_4b.py`                                    |                                                                                                         | 
| Supp Data               | DISK learns meaningful representations of 1 sec-long sequences of the Mouse FL2 dataset                                | -                                              | outputs of DISK main code (`embedding_umap` script)                                                     |
| Supp Data               | Number of parameters for each network.                                                                                 | -                                              | -                                                                                                       |
| Supp Data               | Comparison [...] 2-fish dataset, all keypoints taken together or only 1 fish                                           | `fig_4b.py`                                              | -                                                                                                       | 


## Comparison with other published methods

See [Note file](notes_comparison_other_methods.md) for the description of the comparison and step by step pipeline.

Links to external github repositories:
- [Optipose](https://github.com/mahir1010/OptiPose)
- [Keypoint-moseq](https://github.com/dattalab/keypoint-moseq)
- [MarkerBasedImputation](https://github.com/diegoaldarondo/MarkerBasedImputation)


## *Switch* data augmentation module

- Available in the branch `swap` of DISK (we used "switch" in the manuscript and "swap" in the code, but they mean the same thing. Our apologies)
- 
