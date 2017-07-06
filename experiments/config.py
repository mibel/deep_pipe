from functools import partial
from typing import Iterable

from experiments.datasets.base import make_cached, Dataset
from experiments.dl import Optimizer, Model
from experiments.datasets.config import dataset_name2dataset
from experiments.splitters.config import splitter_name2splitter
from experiments.dl.model_cores.config import model_name2model
from experiments.batch_iterators.config import batch_iter_name2batch_iter

__all__ = ['config_dataset', 'config_splitter', 'config_optimizer',
           'config_model', 'config_batch_iter', 'config_model']

default_config = {
    "dataset": None,
    "dataset_cached": False,
    "dataset__params": {},
    "splitter": None,
    "splitter__params": {},
    "optimizer": "optimizer",
    "optimizer__params": {},
    "model": None,
    "model__params": {},
    "batch_iter_factory": None,
    "batch_iter_factory__params": {},
    "results_path": None,
}

module_type2module_constructor_mapping = {
    'dataset': dataset_name2dataset,
    'splitter': splitter_name2splitter,
    'optimizer': {'optimizer': Optimizer},
    'model': model_name2model,
    'batch_iter': batch_iter_name2batch_iter
}


def config_object(module_type, config, **kwargs):
    name = config[module_type]
    params = config[f'{module_type}__params']

    module = module_type2module_constructor_mapping[module_type][name](
        **params, **kwargs)
    return module


def config_dataset(config) -> Dataset:
    dataset = config_object('dataset', config)
    if config['dataset_cached']:
        dataset = make_cached(dataset)

    return dataset


def config_splitter(config) -> callable:
    return config_object('splitter', config)


def config_optimizer(config) -> Optimizer:
    return config_object('optimizer', config)


def config_model(config, *, optimizer, n_chans_in, n_chans_out) -> Model:
    model_core = config_object('model', config, optimizer=optimizer,
                               n_chans_in=n_chans_in, n_chans_out=n_chans_out)
    return Model(model_core)


def config_batch_iter_factory(config) -> Iterable:
    module_type = 'batch_iter_factory'
    name = config[module_type]
    params = config[f'{module_type}__params']

    batch_iter_factory = partial(
        module_type2module_constructor_mapping[module_type][name],
        **params
    )

    return batch_iter_factory