import gnumpy as gnp

class LayerConfig:
    """Configuration of a layer's settings for learning. Contains the
    following attributes
    - learn_rate
    - momentum
    - weight_decay
    - [To be added: sparsity, drop-out, etc.]
    """
    def __init__(self):
        pass

class LayerStore:
    """An object to store a layer's type of activation function, and the
    weight matrix. List of attributes:
    - W
    - b
    - act_type
    """
    def __init__(self):
        pass

class BaseLayer(object):
    """Base class for layers in a neural net."""
    def __init__(self, in_dim, out_dim, act_type, 
            weight_decay=0, weight_constraint=0, dropout=0):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.act_type = act_type
        self.weight_decay = weight_decay

        # TODO Not implemented yet
        self.weight_constraint = weight_constraint
        self.dropout = dropout

    def init_weight(self, init_scale):
        """Initialize the weights to small normally distributed numbers"""
        # note the weight and the bias are treated separately for memory 
        # efficiency
        self.W = init_scale * gnp.randn(self.in_dim, self.out_dim)
        self.b = init_scale * gnp.randn(1, self.out_dim)
        self.Winc = self.W * 0
        self.binc = self.b * 0

    def load_weight(self, W, b):
        self.W = gnp.garray(W)
        self.b = gnp.garray(b)
        self.Winc = self.W * 0
        self.binc = self.b * 0

class Layer(BaseLayer):
    """A layer in the neural network."""
    def forward(self, Xbelow):
        """Take the input from the layer below and compute the output, and
        store the activation in this layer."""

        self.Xbelow = Xbelow
        self.A = self.Xbelow.dot(self.W) + self.b

        self.Xabove = self.act_type.forward(self.A)

        return self.Xabove

    def gradient(self, dLdXabove):
        """Compute gradient of this layer given the input from the layer 
        above."""
        self.dLdXabove = dLdXabove
        g = self.dLdXabove * self.act_type.derivative(self.Xabove, self.A)

        self.dLdXbelow = g.dot(self.W.T)
        self.dLdW = self.Xbelow.T.dot(g)
        self.dLdb = g.sum(axis=0)

    def backprop(self, dLdXabove, config):
        """Compute the gradients at this layer given the input from the layer
        above."""
        self.gradient(dLdXabove)

        # update W
        Winc = -config.learn_rate * self.dLdW / self.A.shape[0]
        binc = -config.learn_rate * self.dLdb / self.A.shape[0]

        # layer specific weight-decay overwrites the global weight-decay
        if self.weight_decay > 0:
            weight_decay = self.weight_decay
        else:
            weight_decay = config.weight_decay

        if weight_decay > 0:
            Winc -= (config.learn_rate * weight_decay) * self.W
            binc -= (config.learn_rate * weight_decay) * self.b
        if config.momentum > 0:
            Winc += config.momentum * self.Winc
            binc += config.momentum * self.binc

        self.W += Winc
        self.b += binc
        self.Winc = Winc
        self.binc = binc

        return self.dLdXbelow

    def __str__(self):
        s = 'layer ' + str(self.out_dim) + ' ' + self.act_type.name()
        if self.weight_decay > 0:
            s += ' wd %g' % self.weight_decay
        if self.weight_constraint > 0:
            s += ' wc %g' % self.weight_constraint
        if self.dropout > 0:
            s += ' dropout %g' % self.dropout

        return s

class OutputLayer(BaseLayer):
    """The output layer."""
    def init_weight(self, init_scale):
        super(OutputLayer, self).init_weight(init_scale)
        if self.act_type.init_to_zero():
            self.W *= 0
            self.b *= 0

    def forward(self, Xtop):
        """Perform the forward pass, given the top layer output of the net, go
        through the output layer and compute the output activation."""
        self.Xtop = Xtop
        self.A = self.Xtop.dot(self.W) + self.b

    def predict(self):
        """Make predictions based on the computed activation."""
        return self.act_type.predict(self.A)

    def loss(self, Z):
        """Compute the loss of the current prediction compared with the given
        ground truth. This function should be called after forward function."""
        self.Z = Z
        self.Y = self.act_type.output(self.A, Z)
        return self.act_type.loss(self.Y, Z, self.A)

    def task_loss(self, Z, task_loss_fn=None):
        """Compute task loss, as compared to the surrogate loss used for
        training, this can be a more complicated loss."""
        self.Z = Z
        self.Y = self.act_type.output(self.A, Z)
        return self.act_type.task_loss(self.Y, Z, self.A, task_loss_fn)

    def gradient(self, dLdA=None):
        """Compute gradient of this layer given the outputs and ground-truth."""
        if dLdA == None:
            dLdA = self.act_type.derivative(self.Y, self.Z, self.A)
        self.dLdW = self.Xtop.T.dot(dLdA)
        self.dLdXtop = dLdA.dot(self.W.T)
        self.dLdb = dLdA.sum(axis=0)

    def set_gradient(self, ground_truth_Y, inferred_Y, C):
        """Set the gradient using ground_truth_Y and inferred_Y, with the
        gradient weights given in C. This will be used in the semi-supervised
        applications.

        Both ground_truth_Y and inferred_Y should be of size N*K, where N
        is the number of examples in the batch and K is the number of classes.
        C is a N-element vector, assigning a weight for each example.
        """
        self.gradient(gnp.garray((inferred_Y - ground_truth_Y) * 
            C.reshape(inferred_Y.shape[0],1)))

    def backprop(self, config, Z=None):
        """Backprop through the top layer, using the recorded ground truth
        when calling the loss function or supplied when calling this fucntion.
        The weights of this output layer is updated according to the
        configuration specified in config. The graident of the lower layer
        outputs is returned."""
        if Z:
            self.Z = Z

        self.gradient()
        return self.update_weights(config)

    def update_weights(self, config):
        """The gradients, dLdW, dLdb and dLdXtop should be ready before calling
        this method."""
        # update W
        Winc = -config.learn_rate * self.dLdW / self.A.shape[0]
        binc = -config.learn_rate * self.dLdb / self.A.shape[0]

        # layer specific weight-decay overwrites global weight-decay
        if self.weight_decay > 0:
            weight_decay = self.weight_decay
        else:
            weight_decay = config.weight_decay

        if weight_decay > 0:
            Winc -= config.learn_rate * weight_decay * self.W
            binc -= config.learn_rate * weight_decay * self.b
        if config.momentum > 0:
            Winc += config.momentum * self.Winc
            binc += config.momentum * self.binc

        self.W += Winc
        self.b += binc
        self.Winc = Winc
        self.binc = binc

        return self.dLdXtop

    def __str__(self):
        s = 'output ' + str(self.out_dim) + ' ' + self.act_type.name()
        if self.weight_decay > 0:
            s += ' wd %g' % self.weight_decay
        if self.weight_constraint > 0:
            s += ' wc %g' % self.weight_constraint
        if self.dropout > 0:
            s += ' dropout %g' % self.dropout
        return s
