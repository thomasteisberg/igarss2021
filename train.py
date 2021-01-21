import os
import pickle
import yaml
import argparse

import numpy as np

import tensorflow as tf

import wandb
from wandb.keras import WandbCallback

from tqdm.keras import TqdmCallback

from dataset import PINNDataset
from model import PINN
from visualization import GeneratorVisualizationCallback

if __name__ == '__main__':
    #
    # Parameters
    #

    parser = argparse.ArgumentParser()
    parser.add_argument('--params', dest='param_yaml', default='params_2d.yaml',
                        help='path to YAML file with run parameters')

    args = parser.parse_args()

    parameter_yaml_filename = args.param_yaml

    with open(parameter_yaml_filename) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config['params_yaml_path'] = parameter_yaml_filename

    wandb.init(project=config.get('wandb_project', "igarss2021"), config=config)
    wandb.save(parameter_yaml_filename)
    config = wandb.config
    config['wandb'] = True

    #
    # Data Loading
    #

    with open(config['input_data_filename'], 'rb') as f:
        data = pickle.load(f)

    dataset = PINNDataset(data, batch_size=config['batch_size'],
                            mode=config['mode'], n_random=config['n_random_points'])

    #
    # Create model
    #

    model = PINN(config, dataset)

    callbacks = [
            GeneratorVisualizationCallback(config, dataset, model, every_n_epochs=config.get('visualize_every_n_epochs', 1)),
            WandbCallback(),
            TqdmCallback(verbose=0)
        ]

    model.compile()


    model.generator.fit(dataset, verbose=0, epochs=config['epochs'], callbacks=callbacks)

    model.generator.save(os.path.join(wandb.run.dir, "gen_model"))
