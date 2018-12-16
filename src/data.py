import pandas as pd
from typing import Tuple, Callable

import tensorflow as tf

from src.utils.types import path

labels_map = {
    'anger': 0,
    'fear': 1,
    'disgust': 2,
    'surprise': 3,
    'sad': 4,
    'joy': 5,
}


def get_data_fn(src: path) -> Callable[[], Tuple[str, int]]:
    df = pd.read_csv(src, header=None, names=['label', 'sentence'], sep='\t')

    def get_data() -> Tuple[str, int]:
        for _, row in df.iterrows():
            label, sentence = row.tolist()
            sentence = sentence.replace("[#TRIGGERWORD#]", 'OOV')
            sentence = sentence.replace("http://url.removed", 'url')
            sentence = sentence[:200]
            label_idx = labels_map[label]
            yield sentence, label_idx

    return get_data


def one_hot_labels_fn(num_classes: int):
    def one_hot_labels(_: tf.Tensor, label: tf.Tensor):
        return _, tf.one_hot(label, num_classes)

    return one_hot_labels
