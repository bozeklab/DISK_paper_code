from cvkit.pose_estimation.data_readers import FlattenedDataStore
from cvkit_optipose.pose_estimation.processors.filter import SequentialPosturalAutoEncoder

from cvkit.pose_estimation.config import PoseEstimationConfig
import os
import sys
from glob import glob

#######################################################################################################################
## Awaits 3 arguments:
print('ARGS', sys.argv)

# dataset name (matching the config file)
dataset_name = sys.argv[1]
# overlap between sequences, between 0 and sequence_length
overlap = int(sys.argv[2])
# path to model folder
model_folder_path = sys.argv[3]

#test_folder = sys.argv[4] # loaded from config file later
#######################################################################################################################

config_ = PoseEstimationConfig(f'./example_configs/{dataset_name}.yml')
output_dims = 120
n_pcm, n_cm, _, n_heads = os.path.basename(model_folder_path).lstrip('optipose-').split('-')
n_pcm = int(eval(n_pcm))
n_cm = int(eval(n_cm))
n_heads = int(eval(n_heads.split('_')[0] if '_' in n_heads else n_heads))

model = SequentialPosturalAutoEncoder(config_, 60, n_pcm, n_cm, n_heads, overlap=overlap, output_dim=output_dims,
                                      weights=model_folder_path)
model.PRINT = True

output_path = os.path.join(os.path.dirname(model_folder_path), 'test')
if not os.path.exists(output_path):
    os.mkdir(output_path)

for test_file in glob(os.path.join(config_.test_files, '*.csv')):
    pred = FlattenedDataStore(config_.body_parts, test_file)
    model.process(pred)
    pred.save_file(os.path.join(output_path, f'{os.path.basename(pred.base_file_path)}_model_{n_pcm}_{n_cm}_{n_heads}.csv')

