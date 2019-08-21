import tensorflow as tf
from models.layers import Layer


class Dense(Layer):
    """Dense layer."""

    def __init__(self, input_dim, output_dim, dropout=0.,
                 act=tf.nn.relu, placeholders=None, bias=True, featureless=False,
                 sparse_inputs=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        self.dropout = dropout

        self.act = act
        self.featureless = featureless
        self.bias = bias
        self.input_dim = input_dim
        self.output_dim = output_dim

        # helper variable for sparse dropout
        self.sparse_inputs = sparse_inputs
        if sparse_inputs:
            self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['weights'] = tf.compat.v1.get_variable('weights', shape=(input_dim, output_dim),
                                                             dtype=tf.float32,
                                                             initializer=tf.contrib.layers.xavier_initializer(),
                                                             regularizer=tf.contrib.layers.l2_regularizer(FLAGS.weight_decay))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        x = tf.nn.dropout(x, rate=self.dropout)

        # transform
        output = tf.matmul(x, self.vars['weights'])

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)
