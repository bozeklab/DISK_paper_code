from cvkit.pose_estimation.data_readers import FlattenedDataStore
from cvkit_optipose.pose_estimation.processors.filter import SequentialPosturalAutoEncoder

from cvkit.pose_estimation.config import PoseEstimationConfig
import os
import sys
from glob import glob

print('ARGS', sys.argv)
dataset_name = sys.argv[1]
overlap = int(sys.argv[2])
model_folder_path = sys.argv[3]
test_folder = sys.argv[4]

config_ = PoseEstimationConfig(f'./example_configs/{dataset_name}.yml')
output_dims = 120
n_pcm, n_cm, _, n_heads = os.path.basename(model_folder_path).lstrip('optipose-').split('-')
n_pcm = int(eval(n_pcm))
n_cm = int(eval(n_cm))
n_heads = int(eval(n_heads.split('_')[0] if '_' in n_heads else n_heads))

model = SequentialPosturalAutoEncoder(config_, 60, n_pcm, n_cm, n_heads, overlap=overlap, output_dim=output_dims,
                                      weights=model_folder_path)
model.PRINT = True

test_files = glob(os.path.join(test_folder, 'test_*_sample*.csv'))
#test_files = glob(os.path.join(test_folder, 'test_w-all-nans_file*.csv'))
for f in test_files:
    print(f)
    pred = FlattenedDataStore(config_.body_parts, f)
    model.process(pred)
    pred.save_file(os.path.join(test_folder, 'optipose', f'{os.path.basename(f).split(".")[0]}_model_{n_pcm}_{n_cm}_{n_heads}.csv'))


