import tensorflow as tf
from models import models
from models.layers import GraphConvolution
from utils import metrics


class MLP(Model):
    """ A standard multi-layer perceptron """

    def __init__(self, placeholders, dims, categorical=True, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.dims = dims
        self.input_dim = dims[0]
        self.output_dim = dims[-1]
        self.placeholders = placeholders
        self.categorical = categorical

        self.inputs = placeholders['features']
        self.labels = placeholders['labels']

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        if self.categorical:
            self.loss += metrics.masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                              self.placeholders['labels_mask'])
        # L2
        else:
            diff = self.labels - self.outputs
            self.loss += tf.reduce_sum(
                tf.sqrt(tf.reduce_sum(diff * diff, axis=1)))

    def _accuracy(self):
        if self.categorical:
            self.accuracy = metrics.masked_accuracy(self.outputs, self.placeholders['labels'],
                                                    self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(layers.Dense(input_dim=self.input_dim,
                                        output_dim=self.dims[1],
                                        act=tf.nn.relu,
                                        dropout=self.placeholders['dropout'],
                                        sparse_inputs=False,
                                        logging=self.logging))

        self.layers.append(layers.Dense(input_dim=self.dims[1],
                                        output_dim=self.output_dim,
                                        act=lambda x: x,
                                        dropout=self.placeholders['dropout'],
                                        logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)
