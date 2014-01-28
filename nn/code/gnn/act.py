import gnumpy as gnp
import const
import numpy as np


class ActivationTypeError(Exception):
    """Not supported activation function type."""
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        if isinstance(self.msg, str):
            return self.msg
        else:
            return repr(self.msg)

class OutputTypeError(Exception):
    """Not supported output function type."""
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        if isinstance(self.msg, str):
            return self.msg
        else:
            return repr(self.msg)

class LabelValueError(Exception):
    """Incorrect label value for the type of output."""
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        if isinstance(self.msg, str):
            return self.msg
        else:
            return repr(self.msg)


###############################################################################
#                           Intermediate layers
###############################################################################

class ActivationType(object):
    """Base class for an activation type object."""
    def __init__(self):
        pass

    def forward(self, A):
        """Forward pass, compute output from the activation"""
        pass

    def derivative(self, X, A):
        """Backprop pass, compute derivative dX/dA"""
        pass

    def name(self):
        """Name of the activation type."""
        pass

class SigmoidActivation(ActivationType):
    """Logistic sigmoid function."""
    def forward(self, A):
        return gnp.logistic(A)

    def derivative(self, X, A):
        return X * (1 - X)

    def name(self):
        return 'sigmoid'

class TanhActivation(ActivationType):
    """tanh."""
    def forward(self, A):
        return gnp.tanh(A)

    def derivative(self, X, A):
        return 1 - X * X

    def name(self):
        return 'tanh'

class ReluActivation(ActivationType):
    """relu."""
    def forward(self, A):
        # note currently there is no implementation of maximum in gnumpy
        return (A > 0) * A

    def derivative(self, X, A):
        return (A > 0)

    def name(self):
        return 'relu'


class ActivationTypeManager(object):
    """Maintains a list of ActivationType objects."""
    def __init__(self):
        self.sigmoid = SigmoidActivation()
        self.tanh = TanhActivation()
        self.relu = ReluActivation()

        self.name_to_type = {
                self.sigmoid.name() : self.sigmoid,
                self.tanh.name() : self.tanh,
                self.relu.name() : self.relu }

    def get_type_by_name(self, name):
        if name not in self.name_to_type:
            raise ActivationTypeError('Activation type %s not defined.' % name)
        return self.name_to_type[name]

    
###############################################################################
#                               Output layer
###############################################################################

class OutputType(object):
    """Base class for all different output types."""
    def __init__(self):
        pass

    def output(self, A, Z):
        """Compute the output from the activation. For all output units, the 
        activation is always X.dot(W), while for different output units the 
        output can be different, e.g. softmax outputs a distribution. This is
        the output used to compute loss."""
        pass

    def predict(self, A):
        """Making prediction based on the activation.  Usually it is just an 
        argmax for classification problems and identity for regression."""
        pass

    def loss(self, Y, Z, A):
        """Compute the loss by comparing the output Y and the ground truth Z."""
        pass

    def derivative(self, Y, Z, A):
        """Compute the derivative of the output layer, using top layer 
        activation A, output Y and ground truth Z. 
        
        Return a derivative dL/dA."""
        pass

    def name(self):
        """Name of the output type.  A string."""
        pass

    def label_vec_to_mat(self, z, K):
        """Convert a label vector into a matrix for convenience of processing.
        For regression problems, nothing needs to be done in this function.
        
        For classification problems, z should be a N-d vector (a 1-d matrix)."""
        pass

    def init_to_zero(self):
        """Some output types require weights to initialize from 0 rather than 
        from some small random weights. Return whether this is true for this
        type."""
        pass

class LinearOutput(OutputType):
    """linear units"""
    def output(self, A, Z=None):
        return A

    def predict(self, A):
        return A

    def loss(self, Y, Z, A=None):
        # square loss
        return ((Y - Z)**2).sum()/2

    def task_loss(self, Y, Z, A=None, task_loss_fn=None):
        # root mean square error
        if task_loss_fn == None:
            return gnp.sqrt(((Y-Z)**2).mean())
        else:
            return task_loss_fn(Y, Z, A)

    def derivative(self, Y, Z, A):
        return A - Z

    def name(self):
        return 'linear'

    def label_vec_to_mat(self, z, K=None):
        return gnp.garray(z)
        # return z

    def init_to_zero(self):
        return False

class SoftmaxOutput(OutputType):
    """Softmax units"""
    def output(self, A, Z=None):
        # Note gnumpy does not have a expand_dims function
        amax = A.max(axis=1)
        Y = gnp.exp(A - amax.reshape(amax.size,1))
        ysum = Y.sum(axis=1)
        return Y / ysum.reshape(ysum.size,1)

    def predict(self, A):
        return A.argmax(axis=1)

    def loss(self, Y, Z, A=None):
        # cross entropy loss
        return -(Z * gnp.log(Y + const.epsilon)).sum()

    def derivative(self, Y, Z, A):
        return Y - Z

    def name(self):
        return 'softmax'

    def label_vec_to_mat(self, z, K):
        ncases = len(z)

        Z = np.zeros((ncases, K))
        Z[np.arange(ncases), z] = 1

        #Z = gnp.zeros((ncases, K))
        #for i in range(ncases):
        #    k = z[i]
        #    Z[i][k] = 1

        #return Z
        return gnp.garray(Z)

    def init_to_zero(self):
        return False

class HingeOutput(OutputType):
    """Hinge layer for binary classification only."""
    def output(self, A, Z=None):
        return A

    def predict(self, A):
        return (A > 0).reshape(A.size)

    def loss(self, Y, Z, A=None):
        M = 1 - Y*Z
        return ((M > 0) * M).sum()

    def derivative(self, Y, Z, A):
        return -((A * Z < 1) * Z)

    def name(self):
        return 'hinge'

    def label_vec_to_mat(self, z, K):
        if z.max() > 1:
            raise LabelValueError('More than two classes detected! ' \
                    + 'Choose one-versus-all output instead.')

        ncases = len(z)
        Z = np.ones((ncases, 1))
        for i in range(ncases):
            if z[i] == 0:
                Z[i] = -1
            else:
                Z[i] = 1

        return gnp.garray(Z)

    def init_to_zero(self):
        return True

class OneVersusAllOutput(OutputType):
    """One versus all hinge loss output layer."""
    def output(self, A, Z=None):
        return A

    def predict(self, A):
        return A.argmax(axis=1)

    def loss(self, Y, Z, A=None):
        M = 1 - Y * Z
        return ((M > 0) * M).sum()

    def derivative(self, Y, Z, A):
        return -((A * Z < 1) * Z)

    def name(self):
        return 'onevall'

    def label_vec_to_mat(self, z, K):
        ncases = len(z)
        Z = -np.ones((ncases, K))
        for i in range(ncases):
            Z[i][z[i]] = 1
        return gnp.garray(Z)

    def init_to_zero(self):
        return True

class L2svmOutput(OutputType):
    """One versus all L2 hinge loss output layer. This can be easily changed to
    L-n SVM."""
    def output(self, A, Z=None):
        return A

    def predict(self, A):
        return A.argmax(axis=1)

    def loss(self, Y, Z, A=None):
        M = 1 - Y * Z
        return (((M > 0) * M)**2).sum()

    def derivative(self, Y, Z, A):
        M = 1 - A * Z
        return -2 * ((M > 0) * M * Z)

    def name(self):
        return 'l2svm'

    def label_vec_to_mat(self, z, K):
        ncases = len(z)
        Z = -np.ones((ncases, K))
        for i in range(ncases):
            Z[i][z[i]] = 1
        return gnp.garray(Z)

    def init_to_zero(self):
        return True

class LnsvmOutput(OutputType):
    """One versus all L-n hinge loss output layer."""

    # default L2 loss
    n = 2

    def set_n(self, n):
        self.n = n

    def output(self, A, Z=None):
        return A

    def predict(self, A):
        return A.argmax(axis=1)

    def loss(self, Y, Z, A=None):
        M = 1 - Y * Z
        return (((M > 0) * M)**self.n).sum()

    def derivative(self, Y, Z, A):
        M = 1 - A * Z
        absM = M.abs() + const.epsilon
        return -(self.n * (M > 0) * (absM**(self.n-1)) * Z)

    def name(self):
        return 'lnsvm'

    def label_vec_to_mat(self, z, K):
        ncases = len(z)
        Z = -np.ones((ncases, K))
        for i in range(ncases):
            Z[i][z[i]] = 1
        return gnp.garray(Z)

    def init_to_zero(self):
        return True

class LnsvmVariantOutput(OutputType):
    """A variant of one versus all L-n hinge loss output layer.
    l(y,z) = max{0, 1-1/n * yz}^n
    """

    # default L1 loss
    n = 1

    def set_n(self, n):
        self.n = n

    def output(self, A, Z=None):
        return A

    def predict(self, A):
        return A.argmax(axis=1)

    def loss(self, Y, Z, A=None):
        M = 1 - 1.0/self.n * Y * Z
        return (((M > 0) * M)**self.n).sum()

    def derivative(self, Y, Z, A):
        M = 1 - 1.0/self.n * A * Z
        absM = M.abs() + const.epsilon
        return -((M > 0) * (absM**(self.n-1)) * Z)

    def name(self):
        return 'lnsvm_variant'

    def label_vec_to_mat(self, z, K):
        ncases = len(z)
        Z = -np.ones((ncases, K))
        for i in range(ncases):
            Z[i][z[i]] = 1
        return gnp.garray(Z)

class MulticlassHingeOutput(OutputType):
    """Multi-class structured hinge loss layer."""
    def set_loss(self, delta):
        self.delta = gnp.garray(delta)
        self.cpu_delta = delta

    def output(self, A, Z):
        z = Z.argmax(axis=1)
        return self.label_vec_to_mat((self.cpu_delta[z,:] + A.asarray()).argmax(axis=1), Z.shape[1])

    def predict(self, A):
        return A.argmax(axis=1)

    def loss(self, Y, Z, A):
        # this is the structured hinge loss
        y = Y.argmax(axis=1)
        z = Z.argmax(axis=1)

        L = ((Y - Z) * A).sum()

        for i in range(len(y)):
            L += self.cpu_delta[y[i], z[i]]
        return L

    def derivative(self, Y, Z, A):
        return Y - Z

    def name(self):
        return 'multiclass_hinge'

    def label_vec_to_mat(self, z, K):
        ncases = len(z)
        Z = np.zeros((ncases, K))
        for i in range(ncases):
            Z[i][z[i]] = 1
        return gnp.garray(Z)

    def init_to_zero(self):
        return True


class OutputTypeManager:
    """Maintains a list of output type objects."""
    def __init__(self):
        self.linear = LinearOutput()
        self.softmax = SoftmaxOutput()
        self.hinge = HingeOutput()
        self.onevall = OneVersusAllOutput()
        self.l2svm = L2svmOutput()
        self.lnsvm = LnsvmOutput()
        self.lnsvm_variant = LnsvmVariantOutput()
        self.multiclass_hinge = MulticlassHingeOutput()

        self.name_to_type = {
                self.linear.name() : self.linear,
                self.softmax.name() : self.softmax,
                self.hinge.name() : self.hinge,
                self.onevall.name() : self.onevall,
                self.l2svm.name() : self.l2svm,
                self.lnsvm.name() : self.lnsvm,
                self.lnsvm_variant.name() : self.lnsvm_variant,
                self.multiclass_hinge.name() : self.multiclass_hinge }

    def get_output_by_name(self, name):
        if name not in self.name_to_type:
            raise OutputTypeError('Output type %s not defined.' % name)
        return self.name_to_type[name]


