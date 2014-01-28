import numpy as np

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
    - act_type
    """
    def __init__(self):
        pass

class BaseLayer:
    """Base class for layers in a neural net."""
    def __init__(self, in_dim, out_dim, act_type):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.act_type = act_type

    def init_weight(self, init_scale):
        """Initialize the weights to small normally distributed numbers"""

        # note the weight matrix is padded with a column for biases
        self.W = init_scale * np.random.randn(self.in_dim + 1, self.out_dim)
        self.Winc = self.W * 0

class Layer(BaseLayer):
    """A layer in the neural network."""
    def forward(self, Xbelow):
        """Take the input from the layer below and compute the output, and
        store the activation in this layer."""

        # pad with an extra column of 1's - for the biases
        self.Xbelow = np.c_[Xbelow, np.ones((Xbelow.shape[0],1))]
        self.A = self.Xbelow.dot(self.W)

        self.Xabove = self.act_type.forward(self.A)

        return self.Xabove

    def gradient(self, dLdXabove):
        """Compute gradient of this layer given the input from the layer 
        above."""
        self.dXdA = self.act_type.derivative(self.Xabove, self.A)

        self.dLdXabove = dLdXabove
        g = self.dLdXabove * self.dXdA

        self.dLdXbelow = g.dot(self.W.T)
        self.dLdW = self.Xbelow.T.dot(g)

    def backprop(self, dLdXabove, config):
        """Compute the gradients at this layer given the input from the layer
        above."""
        self.gradient(dLdXabove)

        # update W
        Winc = -config.learn_rate * self.dLdW / self.A.shape[0]
        if config.weight_decay > 0:
            Winc -= config.learn_rate * config.weight_decay * self.W
        if config.momentum > 0:
            Winc += config.momentum * self.Winc

        self.W += Winc
        self.Winc = Winc

        # don't return the bias part
        return self.dLdXbelow[:,:-1]

    def __str__(self):
        return 'layer ' + str(self.out_dim) + ' ' + self.act_type.name()


class OutputLayer(BaseLayer):
    """The output layer."""
    def forward(self, Xtop):
        """Perform the forward pass, given the top layer output of the net, go
        through the output layer and compute the output activation."""
        self.Xtop = np.c_[Xtop, np.ones((Xtop.shape[0],1))]
        self.A = self.Xtop.dot(self.W)

    def predict(self):
        """Make predictions based on the computed activation."""
        return self.act_type.predict(self.A)

    def loss(self, Z):
        """Compute the loss of the current prediction compared with the given
        ground truth. This function should be called after forward function."""
        self.Z = Z
        self.Y = self.act_type.output(self.A, Z)
        return self.act_type.loss(self.Y, Z, self.A)

    def gradient(self):
        """Compute gradient of this layer given the outputs and ground-truth."""
        dLdA = self.act_type.derivative(self.Y, self.Z, self.A)
        self.dLdW = self.Xtop.T.dot(dLdA)
        self.dLdXtop = dLdA.dot(self.W.T)

    def backprop(self, config, Z=None):
        """Backprop through the top layer, using the recorded ground truth
        when calling the loss function or supplied when calling this fucntion.
        The weights of this output layer is updated according to the
        configuration specified in config. The graident of the lower layer
        outputs is returned."""
        if Z:
            self.Z = Z

        self.gradient()

        # update W
        Winc = -config.learn_rate * self.dLdW / self.A.shape[0]
        if config.weight_decay > 0:
            Winc -= config.learn_rate * config.weight_decay * self.W
        if config.momentum > 0:
            Winc += config.momentum * self.Winc

        self.W += Winc
        self.Winc = Winc

        # don't return bias part
        return self.dLdXtop[:,:-1]

    def __str__(self):
        return 'output ' + str(self.out_dim) + ' ' + self.act_type.name()

