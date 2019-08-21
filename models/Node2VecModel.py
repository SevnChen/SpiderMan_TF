import tensorflow as tf
from models.models import GeneralizedModel


class Node2VecModel(GeneralizedModel):
    def __init__(self, placeholders, dict_size, degrees, name=None,
                 nodevec_dim=50, lr=0.001, **kwargs):
        """ Simple version of Node2Vec/DeepWalk algorithm.

        Args:
            dict_size: the total number of nodes.
            degrees: numpy array of node degrees, ordered as in the data's id_map
            nodevec_dim: dimension of the vector representation of node.
            lr: learning rate of optimizer.
        """

        super(Node2VecModel, self).__init__(**kwargs)

        self.placeholders = placeholders
        self.degrees = degrees
        self.inputs1 = placeholders["batch1"]
        self.inputs2 = placeholders["batch2"]

        self.batch_size = placeholders['batch_size']
        self.hidden_dim = nodevec_dim

        # following the tensorflow word2vec tutorial
        self.target_embeds = tf.Variable(
            tf.random_uniform([dict_size, nodevec_dim], -1, 1),
            name="target_embeds")
        self.context_embeds = tf.Variable(
            tf.truncated_normal([dict_size, nodevec_dim],
                                stddev=1.0 / math.sqrt(nodevec_dim)),
            name="context_embeds")
        self.context_bias = tf.Variable(
            tf.zeros([dict_size]),
            name="context_bias")

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)

        self.build()

    def _build(self):
        labels = tf.reshape(
            tf.cast(self.placeholders['batch2'], dtype=tf.int64),
            [self.batch_size, 1])
        self.neg_samples, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
            true_classes=labels,
            num_true=1,
            num_sampled=FLAGS.neg_sample_size,
            unique=True,
            range_max=len(self.degrees),
            distortion=0.75,
            unigrams=self.degrees.tolist()))

        self.outputs1 = tf.nn.embedding_lookup(
            self.target_embeds, self.inputs1)
        self.outputs2 = tf.nn.embedding_lookup(
            self.context_embeds, self.inputs2)
        self.outputs2_bias = tf.nn.embedding_lookup(
            self.context_bias, self.inputs2)
        self.neg_outputs = tf.nn.embedding_lookup(
            self.context_embeds, self.neg_samples)
        self.neg_outputs_bias = tf.nn.embedding_lookup(
            self.context_bias, self.neg_samples)

        self.link_pred_layer = BipartiteEdgePredLayer(self.hidden_dim, self.hidden_dim,
                                                      self.placeholders, bilinear_weights=False)

    def build(self):
        self._build()
        # TF graph management
        self._loss()
        self._minimize()
        self._accuracy()

    def _minimize(self):
        self.opt_op = self.optimizer.minimize(self.loss)

    def _loss(self):
        aff = tf.reduce_sum(tf.multiply(
            self.outputs1, self.outputs2), 1) + self.outputs2_bias
        neg_aff = tf.matmul(self.outputs1, tf.transpose(
            self.neg_outputs)) + self.neg_outputs_bias
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(aff), logits=aff)
        negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(neg_aff), logits=neg_aff)
        loss = tf.reduce_sum(true_xent) + tf.reduce_sum(negative_xent)
        self.loss = loss / tf.cast(self.batch_size, tf.float32)
        tf.summary.scalar('loss', self.loss)

    def _accuracy(self):
        # shape: [batch_size]
        aff = self.link_pred_layer.affinity(self.outputs1, self.outputs2)
       # shape : [batch_size x num_neg_samples]
        self.neg_aff = self.link_pred_layer.neg_cost(
            self.outputs1, self.neg_outputs)
        self.neg_aff = tf.reshape(
            self.neg_aff, [self.batch_size, FLAGS.neg_sample_size])
        _aff = tf.expand_dims(aff, axis=1)
        self.aff_all = tf.concat(axis=1, values=[self.neg_aff, _aff])
        size = tf.shape(self.aff_all)[1]
        _, indices_of_ranks = tf.nn.top_k(self.aff_all, k=size)
        _, self.ranks = tf.nn.top_k(-indices_of_ranks, k=size)
        self.mrr = tf.reduce_mean(
            tf.div(1.0, tf.cast(self.ranks[:, -1] + 1, tf.float32)))
        tf.summary.scalar('mrr', self.mrr)
