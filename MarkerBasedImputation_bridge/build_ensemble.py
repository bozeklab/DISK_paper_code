"""Build a model ensemble."""
# import clize
# from keras.models import Model, load_model
# from keras.layers import Input, Lambda, concatenate

import os
import logging
from models import Wave_net
from utils import create_run_folders
import torch
import json
from glob import glob

import matplotlib
if os.uname().nodename == 'france-XPS':
    matplotlib.use('TkAgg')
    basedir = '/home/france/Mounted_dir'
else:
    matplotlib.use('Agg')
    basedir = '/projects/ag-bozek/france'


class EnsembleModel(torch.nn.Module):
    def __init__(self, model_paths, device=torch.device('cpu'), **kwargs):
        """Build an ensemble of models that output the median of all members.
        Not really a model because does not learn anything

        :param model_paths: List of paths to pytorch saved models to include in the ensemble. Currently,
                       requires the same output shape.
        :param device: to put the models on selected device

        """
        super(EnsembleModel, self).__init__()
        self.models = torch.nn.ModuleList()

        model_paths = [mp if mp[0] == '/' else os.path.join(basedir, mp) for mp in model_paths]

        for i in range(len(model_paths)):
            logging.info(f' Loading model from {os.path.join(os.path.dirname(model_paths[i]), "training_info.json")}')

            with open(os.path.join(os.path.dirname(model_paths[i]), "training_info.json"), 'r') as fp:
                self.models_dict_training = json.load(fp)
            model = Wave_net(device=device, **self.models_dict_training)
            checkpoint = torch.load(model_paths[i], map_location=torch.device(device))

            if 'model_state_dict' in checkpoint.keys():
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            model.eval()
            self.models.append(model)

        self.n_members = len(self.models)
        self.input_length = self.models[0].input_length
        self.output_length = self.models[0].output_length


    def forward(self, x):
        outputs = [model(x) for model in self.models]
        member_predictions = torch.cat(outputs, dim=1)
        ensemble_pred = torch.quantile(member_predictions, 0.5, dim=1)
        return ensemble_pred[:, None, :], member_predictions[:, :, None]



def build_ensemble(base_output_path, models_in_ensemble,
                   run_name=None, clean=False,
                   device=torch.device('cpu')):
    """Build an ensemble of models for marker prediction.

    :param base_output_path: Path to base models directory
    :param models_in_ensemble: List of all the paths to the models to be included in the
                               build_ensemble
    :param run_name: Name of the model run (str, default: None)
    :param clean: If True, deletes the contents of the run output path
    """

    # Build the ensemble
    model_ensemble = EnsembleModel(models_in_ensemble, device=device)

    # Build ensemble folder name if needed
    if run_name is None:
        run_name = "model_ensemble"
    logging.info(f"run_name: {run_name}")

    # Initialize run directories
    logging.info('Building run folders')
    run_path = create_run_folders(run_name, base_path=base_output_path, clean=clean)

    # Convert list of models to objects for .mat saving
    model_paths = []
    for i in range(len(models_in_ensemble)):
        model_paths.append(models_in_ensemble[i])

    # Save the training information in a mat file.
    logging.info(f'Saving training info in{os.path.join(run_path, "training_info.json")}')
    with open(os.path.join(run_path, "training_info.json"), "w") as fp:
        json.dump({"base_output_path": base_output_path[len(basedir):],
                        "run_name": run_name,
                        "clean": clean,
                        "model_paths": [mp[len(basedir):].lstrip('/') for mp in model_paths],
                        "n_members": len(models_in_ensemble),
                        "input_length": model_ensemble.models_dict_training["input_length"],
                        "output_length": model_ensemble.models_dict_training["output_length"],},
                  fp)

    logging.info(f'Saving model ensemble in {os.path.join(run_path, "final_model.h5")}')
    torch.save(model_ensemble.state_dict(), os.path.join(run_path, "final_model.h5"))
    return run_path

