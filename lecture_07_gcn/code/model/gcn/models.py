from layers import *
from metrics import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


# 根据Layer来建立Model,主要是设置了self.layers 和 self.activations 建立序列模型，
# 还有init中的其他比如loss、accuracy、optimizer、opt_op等。
class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        # 在子类中可以看出，通过_build方法append各个层
        # 保存每一个layer
        self.layers = []
        # 保存每一次的输入，以及最后一层的输出
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    # 定义私有方法，只能被类中的函数调用，不能在类外单独调用
    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)

        # 以一个两层GCN层为例，输入inputs是features
        # self.activations.append(self.inputs)初始化第一个元素为inputs，也就是features
        # 第一层，hidden=layer(self.activations[-1])，即hidden等于inputs的输出outputs，
        # 并将第一层的输出hidden=outputs加入到activations中
        # 同理，对第二层，hidden作为一个中间存储结果。
        # 最后activations分别存储了三个元素：第一层的输入，第二层的输入（第一层的输出），第二层的输出
        # 最后self.outputs=最后一层的输出
        for layer in self.layers:
            # Layer类重写了__call__ 函数，可以把对象当函数调用,__call__输入为inputs，输出为outputs
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


# 继承Model的多层感知机，主要是重写了基类中没有实现的函数；计算了网络第一层的权重衰减L2损失
# 因为这是半监督学习，还计算了掩码交叉熵masked_softmax_cross_entropy
class MLP(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


# 继承Model的卷机模型 GCN
class GCN(Model):
    def __init__(self, placeholders, input_dim, multi_label=False, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.multi_label = multi_label
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'],
                                                  self.multi_label)

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'],
                                        self.multi_label)
        if not self.multi_label:
            self.pred = tf.argmax(self.outputs, 1)
            self.labels = tf.argmax(self.placeholders['labels'], 1)
        else:
            self.pred = tf.where(tf.nn.sigmoid(self.outputs) >= 0.5, tf.ones(tf.shape(self.outputs)),
                                 tf.zeros(tf.shape(self.outputs)))
            self.pred = tf.cast(self.pred, tf.int32)
            self.labels = self.placeholders['labels']

    def _build(self):
        # 第一层的输入维度：input_dim
        # 第一层的输出维度：output_dim=FLAGS.hidden1=16
        # 第一层的激活函数：relu
        # 两个卷积层
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            featureless=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        # 第二层的输入等于第一层的输出维度：input_dim=FLAGS.hidden1=16
        # 第二层的输出维度：output_dim=placeholders['labels'].get_shape().as_list()[1]=95
        # 第二层的激活函数：lambda x: x，即没有加激活函数
        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,  #
                                            dropout=True,
                                            logging=self.logging))

    def predict(self):
        if not self.multi_label:
            return tf.nn.softmax(self.outputs)
        else:
            return tf.nn.sigmoid(self.outputs)
