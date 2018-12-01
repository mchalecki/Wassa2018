import logging
from typing import Optional

from paths import OUTPUTS, DATA
from src.data import dummy_generator, one_hot_labels_fn
from src.utils.types import path

import tensorflow as tf

log = logging.getLogger(__name__)


class NetworkTrainer:
    def __init__(self):
        self.params = Params()
        self.config = Config()

    def train(self, train_path: path, test_path: path, model_name: Optional[str] = None):
        if model_name:
            self.config.checkpoint_path = self.config.checkpoint_path / model_name
        self.config.checkpoint_path.mkdir(exist_ok=True, parents=True)
        train_spec = tf.estimator.TrainSpec(self.create_input_fn(train_path), max_steps=1000)
        eval_spec = tf.estimator.EvalSpec(self.create_input_fn(test_path))
        # estimator = tf.estimator.Estimator(model_fn, model_dir, run_config)
        # tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    def export(self):
        pass

    def create_input_fn(self, data_path: path):
        def input_fn():
            dataset = (tf.data.Dataset.from_generator(dummy_generator,
                                                      output_types=(tf.string, tf.uint8))
                       .map(one_hot_labels_fn(self.params.num_classes), 4)
                       .shuffle(buffer_size=100)
                       .batch(self.params.batch_size)
                       .repeat())
            return dataset

        return input_fn


class Params:
    def __init__(self):
        self.num_classes = 5
        self.learning_rate = 1e-3
        self.batch_size = 32
        self.max_steps = 10e4


class Config:
    def __init__(self):
        self.checkpoint_path = OUTPUTS.path
        self.save_checkpoint_steps = 1000
        self.save_summary_steps = 100
