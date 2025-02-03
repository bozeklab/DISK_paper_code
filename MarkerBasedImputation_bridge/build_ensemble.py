"""Build a model ensemble."""
# import clize
# from keras.models import Model, load_model
# from keras.layers import Input, Lambda, concatenate
import numpy as np
import os
from scipy.io import savemat
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
    def __init__(self, model_paths, base_output_path, device=torch.device('cpu'), **kwargs):
        """Build an ensemble of models that output the median of all members.
        Not really a model because does not learn anything
        Note: Does not compile the ensemble.
        :param models: List of keras models to include in the ensemble. Curently
                       requires the same output shape.
        :param return_member_data: If True, model will have two outputs: the
                                   ensemble prediction and all member predictions.
                                   Otherwise, the model will output only the
                                   ensemble predictions.
        """
        super(EnsembleModel, self).__init__()
        self.models = []

        model_paths = [mp if mp[0] == '/' else os.path.join(basedir, mp) for mp in model_paths]

        for i in range(len(model_paths)):
            logging.info(f' Loading model from {os.path.join(os.path.dirname(model_paths[i]), "training_info.json")}')
            with open(os.path.join(os.path.dirname(model_paths[i]), "training_info.json"), 'r') as fp:
                self.models_dict_training = json.load(fp)
            model = Wave_net(device=device, **self.models_dict_training)
            model.load_state_dict(torch.load(model_paths[i]))
            model.eval()
            self.models.append(model)
            # models[i] = load_model(os.path.join(base_output_path,
            #                                     models_in_ensemble[i]))
            # models[i].name = 'model_%d' % (i)
        # def ens_median(x):
        #     # return tf.contrib.distributions.percentile(x, 50, axis=1)
        #     return torch.quantile(x, 50, dim=1)
        #
        # def pad(x):
        #     return x[:, None, :]
        #
        # Get outputs from the ensemble models, compute the median, and fix the
        # shape.
        self.n_members = len(self.models)
        # member_predictions = concatenate(outputs, axis=1)
        # ensemble_prediction = Lambda(ens_median)(member_predictions)
        # ensemble_prediction = Lambda(pad)(ensemble_prediction)

        # Return model. No compilation is necessary since there are no additional
        # trainable parameters.
        # if return_member_data:
        #     model = Model(model_input,
        #                   outputs=[ensemble_prediction, member_predictions],
        #                   name='ensemble')
        # else:
        #     model = Model(model_input, ensemble_prediction, name='ensemble')

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
    :param models_in_ensemble: List of all of the models to be included in the
                               build_ensemble
    :param return_member_data: If True, model will have two outputs: the
                               ensemble prediction and all member predictions.
    :param run_name: Name of the model run
    :param clean: If True, deletes the contents of the run output path
    """
    # Load all of the models to be used as members of the ensemble


    # Build the ensemble
    model_ensemble = EnsembleModel(models_in_ensemble, base_output_path)

    # Build ensemble folder name if needed
    if run_name is None:
        run_name = "model_ensemble"
    logging.info(f"run_name: {run_name}")

    # Initialize run directories
    logging.info('Building run folders')
    run_path = create_run_folders(run_name, base_path=base_output_path,
                                  clean=clean)

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

if __name__ == '__main__':

    models = glob(os.path.join(basedir, 'results_behavior/MarkerBasedImputation_run/models-wave_net_epochs=30_input_9_output_1*/best_model.h5'))
    device = torch.device('cuda:0')

    build_ensemble(os.path.join(basedir, 'results_behavior/MarkerBasedImputation/'),
                   models, run_name=None, clean=False, device=device)