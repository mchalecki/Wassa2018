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

            with tf.name_scope("metrics"):
                y_true = tf.argmax(labels, -1)
                y_pred = tf.argmax(self.prediction, 1)

                self.acc = tf.metrics.accuracy(labels=y_true,
                                               predictions=y_pred)
                tf.summary.scalar('accuracy', self.acc[1])

            with tf.name_scope("training"):
                step = tf.train.get_global_step()
                self.train_op = tf.train.AdamOptimizer(self.params.learning_rate).minimize(self.loss, step)


class Network(tf.keras.Model):
    def __init__(self, num_classes: int, training: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes

        self.elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=training)
        self.drop = tf.keras.layers.Dropout(.4)
        self.net1 = tf.keras.Sequential([
            tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(1024, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(512, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(512)),
        ])
        self.net2 = tf.keras.Sequential([
            tf.keras.layers.Dense(128),
            tf.keras.layers.Dense(self.num_classes)
        ])

    def __call__(self, inputs, training=False, **kwargs):
        embeddings = self.elmo(inputs, signature="default", as_dict=True)["elmo"]
        out = self.drop(embeddings, training=training)
        out = self.net1(out)
        out = self.drop(out, training=training)
        out = self.net2(out)
        return out
