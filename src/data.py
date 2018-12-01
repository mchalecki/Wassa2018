import random
from typing import Tuple

import tensorflow as tf

dummy_sentences = [
    ("This is test sentence.", 0),
    ("Test sentences are to test model", 0),
    ("Thos sentence is for testing purposes", 0),
    ("Sky is blue.", 1),
    ("The are clouds up.", 1),
    ("I hope it wont rain.", 1),
    ("That's what she said!", 2),
    ("It is what she was telling me.", 2),
    ("Susan acknowledged that", 2)
]


def dummy_generator() -> Tuple[str, int]:
    """
    Return string sentence and label as int.
    :param maxlen:
    :param categories:
    :return:
    """
    yield random.choice(dummy_sentences)


def one_hot_labels_fn(num_classes: int):
    def one_hot_labels(_: tf.Tensor, label: tf.Tensor):
        return _, tf.one_hot(label, num_classes)

    return one_hot_labels
