import sys

import tensorflow as tf
import tensorflow_hub as hub


class Model:
    def __init__(self, features, labels, params, mode):
        self.params = params

        self.raw_content_input = tf.identity(features, name='raw_input')
        self.is_train = mode == tf.estimator.ModeKeys.TRAIN
        network = Network(self.params.num_classes, training=self.is_train)
        self.prediction = tf.identity(network(self.raw_content_input, training=self.is_train), name='prediction')
        if mode != tf.estimator.ModeKeys.PREDICT:
            with tf.name_scope("loss"):
                self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(labels, self.prediction))
                # tf.summary.scalar('training loss', self.loss)

            with tf.name_scope("training"):
                step = tf.train.get_global_step()
                self.train_op = tf.train.AdamOptimizer(self.params.learning_rate).minimize(self.loss, step)


class Network(tf.keras.Model):
    def __init__(self, num_classes: int, training: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=training)

        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(self.num_classes)
        ])

    def __call__(self, inputs, training=False, **kwargs):
        embeddings = self.elmo(inputs, signature="default", as_dict=True)["default"]
        out = self.net(embeddings)
        return out
