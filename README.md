# DISK_paper_code
Additional scripts to reproduce analyses from the DISK paper.

## List of figures and related code

| Figure                                | Title                                                                                                                  | Script files     | Comments                                                  |
|---------------------------------------|------------------------------------------------------------------------------------------------------------------------|------------------|-----------------------------------------------------------|
| Fig. 1                                | The missing data problem                                                                                               | -                | Measure % missing per dataset                             |
| Fig. 2a                               | DISK and other imputation methods’ performance across datasets                                                         | `fig2_barplots`  | -                                                         |
| Fig. 2 b-j                            | DISK and other imputation methods’ performance across datasets                                                         | -                | outputs of DISK main code (`test_fillmissing` script)     |
| Fig. 3                                | Estimated imputation error                                                                                             | -                | outputs of DISK main code (`test_fillmissing` script)     |
| Fig. 4 a                              | Inference of multiple simultaneously missing keypoints                                                                 | `fig4_n_missing` |                                                           | 
| Fig. 4 c                              | Inference of multiple simultaneously missing keypoints                                                                 | -                |                                                           | 
| Fig. 4 d-f                            | Inference of multiple simultaneously missing keypoints                                                                 | -                | outputs of DISK main code (`test_fillmissing` script)     |
| Fig. 5                                | DISK learns meaningful representations of sequences from the Human dataset                                             |                  |                                                           |
| Fig. 6                                | DISK learns meaningful representations of sequences from the 2-Mice-2D dataset                                         |                  |                                                           |
| Fig. 7 b                              | DISK allows to detect more steps and emphasizes differences in step kinematics between different treatments            |                  |                                                           |
| Fig. 7 c                              | DISK allows to detect more steps and emphasizes differences in step kinematics between different treatments            |                  |                                                           |
| Fig. 7 d-f                            | DISK allows to detect more steps and emphasizes differences in step kinematics between different treatments            |                  |                                                           |
| Fig. B1                               | Comparison of different architectures according to the MPJPE (a) and PCK@0.01 (b) metrics.                             | `fig2_barplots`  | -                                                         |
| Fig. B2                               | DISK performance dependence on dataset size and input length (2-Fish dataset)                                          | `fig_b2.py`      |                                                           | 
| Fig. B3                               | Comparison of DISK, Optipose and Keypoint-MoSeq (Mouse FL2 dataset)                                                    |                  |                                                           |
| Fig. B4                               | Imputation of switched keypoints                                                                                       |                  |                                                           | 
| Fig. B5                               | Test RMSE with respect to the gap length for the other datasets                                                        |                  |                                                           | 
| Fig. B6                               | Estimated error correlation plots [...] for DISK-proba and GRU-proba for all tested datasets                           | -                | outputs of DISK main code (`test_fillmissing` script)     | 
| Fig. B7                               | Observed missing proportions for each keypoint in mouse FL2 and CLB datasets                                           | -                | outputs of DISK main code (`create_proba_missing` script) | 
| Fig. B8 | RMSE wrt the distance between the two fish and the number and scheme of missing keypoints for short gaps and long gaps |                  |                                                           | 
| Fig. B9 | DISK learns meaningful representations of 1 sec-long sequences of the Mouse FL2 dataset                                |                  |                                                           |
| Fig. B10-B16 | Full plots from Fig. 2 b-j                                                                                             | -                | outputs of DISK main code (`create_proba_missing` script) |
| Table B1 | Number of parameters for each network                                                                                  | -                | -                                                         |
| Table B2 | Comparison [...] 2-fish dataset, all keypoints taken together or only 1 fish                                           |                  |                                                           | 


## Comparison with other published methods

See [Note file](notes_comparison_other_methods.md) for the description of the comparison and step by step pipeline.

Links to external github repositories:
- [Optipose](https://github.com/mahir1010/OptiPose)
- [Keypoint-moseq](https://github.com/dattalab/keypoint-moseq)
- [MarkerBasedImputation](https://github.com/diegoaldarondo/MarkerBasedImputation)



