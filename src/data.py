import random
import string
from typing import Tuple

import tensorflow as tf


def dummy_generator(maxlen=20, categories=5) -> Tuple[str, int]:
    """
    Return string sentence and label as int.
    :param maxlen:
    :param categories:
    :return:
    """
    yield (''.join(random.choice(string.ascii_letters + string.digits) for _ in range(random.randint(0, maxlen))),
           random.randint(0, categories)
           )


def one_hot_labels_fn(num_classes: int):
    def one_hot_labels(_: tf.Tensor, label: tf.Tensor):
        return _, tf.one_hot(label, num_classes)

    return one_hot_labels
