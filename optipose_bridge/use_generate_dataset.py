#### To use with optipose conda env

import os
from glob import glob
from cvkit.pose_estimation.data_readers import FlattenedDataStore

from cvkit.pose_estimation.config import PoseEstimationConfig
from OptiPose.model.dataset.generate_dataset import generate_dataset


for is_val in [True, False]:
    for name, namefolder in [
        # ['FL2', 'INH_FL2_keypoints_1_60_wresiduals_w1nan_stride0.5_new'],
        #                      ['CLB', 'INH_CLB_keypoints_1_60_stride0.5'],
        #                      ['DANNCE', 'DANNCE_seq_keypoints_60_stride30_fill10_new'],
        #                      ['MABE', 'MABE_task1_60stride60'],
                             ['Mocap', 'Mocap_keypoints_60_stride30_new'],
                             # ['DF3D', 'DF3D_keypoints_60stride5_new'],
                             # ['Fish', 'Fish_v3_60stride120']
    ]:

        # these .yaml files need to be written for each dataset
        config = PoseEstimationConfig(f'./example_configs/{name}.yml')

        # these individual files are generated with the `create_csv_for_optipose` script
        base_folder = f'/projects/ag-bozek/france/results_behavior/datasets/{namefolder}'
        if is_val:
            csv_files = glob(os.path.join(base_folder, 'for_optipose', 'val', '*.csv'))
            suffix = 'val'
            n_samples = 2000
        else:
            csv_files = glob(os.path.join(base_folder, 'for_optipose', 'train', '*.csv'))
            suffix = 'train'
            n_samples = 20000
        data_stores = [FlattenedDataStore(config.body_parts, os.path.join(base_folder, path)) for path in csv_files]

        #generate_dataset is a function of Optipose
        generate_dataset(os.path.join(base_folder, 'for_optipose'), name, is_val, data_stores, n_samples, min_seq_length=60,
                     max_seq_length=60, min_x=-700,
                     max_x=700, min_y=-700, max_y=700, prefix="", suffix=suffix, random_noise=50)
