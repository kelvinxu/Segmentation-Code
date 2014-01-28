from layer import Layer, OutputLayer, LayerConfig, LayerStore
import numpy as np
import cPickle as pickle
import time
import act

class Data:
    """A container for data objects. Has three attributes, X, T and K."""
    def __init__(self):
        pass

class NN:
    """A class for general purpose neural networks, trained with
    backpropagation. The type of activation functions, number of hidden layers
    and number of units in each layer, the output function, and other options 
    during training can be configured."""
    def __init__(self):
        pass

    def init_net(self, config):
        """config is an instance of class Config"""
        
        import os

        self.config = config

        if config.is_output and (not os.path.exists(config.output_dir)):
            os.makedirs(config.output_dir)

        self.train_data = self.read_data(config.train_data_file)

        if config.is_val:
            self.val_data = self.read_data(config.val_data_file)
        if config.is_test:
            self.test_data = self.read_data(config.test_data_file)

        [num_total_cases, input_dim] = self.train_data.X.shape
        self.num_total_cases = num_total_cases
        self.input_dim = input_dim

        self.num_minibatches = num_total_cases / config.minibatch_size
        if self.num_minibatches < 1:
            self.num_minibatches = 1

        # initialize the network
        self.num_layers = config.num_layers
        self.layer = []
        in_dim = input_dim
        for i in range(0, self.num_layers):
            self.layer.append(Layer(
                in_dim, config.layer[i].out_dim, config.layer[i].act_type))
            in_dim = config.layer[i].out_dim

        self.output = OutputLayer(in_dim, config.output.out_dim,
                config.output.output_type)

        # To use multi-class hinge output, we need to specify the loss function
        if isinstance(self.output.act_type, act.MulticlassHingeOutput):
            if config.loss_file != None:
                self.output.act_type.set_loss(self.read_loss(config.loss_file))
            else:
                self.output.act_type.set_loss(1 - np.eye(self.train_data.K))

        # initialize the weights in every layer
        self._init_weights(config.init_scale, config.random_seed)

    def _init_weights(self, init_scale, random_seed=None):
        if random_seed:
            np.random.seed(random_seed)

        for i in range(0, self.num_layers):
            self.layer[i].init_weight(init_scale)

        self.output.init_weight(init_scale)

    def train(self):
        config = self.config

        # convert t into a matrix in 1-of-K representation if it is a vector
        t = self.train_data.T
        if not self.config.is_regression:
            T_matrix = self.output.act_type.label_vec_to_mat(t, self.train_data.K)
        else:
            T_matrix = t

        layer_config = LayerConfig()
        layer_config.learn_rate = config.learn_rate
        layer_config.momentum = config.momentum
        layer_config.weight_decay = config.weight_decay

        nnstore = NNStore()
        nnstore.init_from_net(self)

        self.display_training_info(-1, 0, 0)
        t_start = time.time()

        for epoch in range(0, config.num_epochs):
            # shuffle the dataset 
            idx = np.random.permutation(self.num_total_cases)
            train_X = self.train_data.X[idx]
            train_T = T_matrix[idx]

            loss = 0

            for batch in range(0, self.num_minibatches):
                i_start = batch * config.minibatch_size
                if not batch == self.num_minibatches - 1:
                    i_end = i_start + config.minibatch_size
                else:
                    i_end = self.num_total_cases

                X = train_X[i_start:i_end]
                T = train_T[i_start:i_end]
                Xbelow = X

                # forward pass
                for i in range(0, self.num_layers):
                    Xbelow = self.layer[i].forward(Xbelow)
                self.output.forward(Xbelow)

                # compute loss
                loss += self.output.loss(T)

                # backprop
                dLdXabove = self.output.backprop(layer_config)
                for i in range(self.num_layers-1, -1, -1):
                    dLdXabove = self.layer[i].backprop(dLdXabove, layer_config)

            # statistics
            avg_loss = 1.0 * loss / self.num_total_cases

            if (epoch + 1) % config.epoch_to_display == 0:
                self.display_training_info(epoch, avg_loss, time.time() - t_start)
                t_start = time.time()

            if (epoch + 1) % config.epoch_to_save == 0:
                nnstore.update_from_net(self)
                nnstore.write(config.output_dir + '/m' + str(epoch + 1) + '.pdata')

    def display_training_info(self, epoch, loss, time):
        """Print training information. Use the config information to determine
        what information to display."""
        if self.config.is_val:
            if self.config.is_test:
                self._display_training_info(epoch, loss, time,
                        self.train_data.X, self.train_data.T,
                        val_data=self.val_data.X, val_labels=self.val_data.T,
                        test_data=self.test_data.X, test_labels=self.test_data.T)
            else:
                self._display_training_info(epoch, loss, time,
                        self.train_data.X, self.train_data.T,
                        val_data=self.val_data.X, val_labels=self.val_data.T)
        else:
            if self.config.is_test:
                self._display_training_info(epoch, loss, time,
                        self.train_data.X, self.train_data.T,
                        test_data=self.test_data.X, test_labels=self.test_data.T)
            else:
                self._display_training_info(epoch, loss, time,
                        self.train_data.X, self.train_data.T)

    def _display_training_info(self, epoch, loss, time, 
            train_data, train_labels, val_data=None, val_labels=None, 
            test_data=None, test_labels=None):
        """Print training information during training."""
        print 'epoch %d, loss %.4f,' % (epoch + 1, loss),
        
        # print loss if it is a regression problem
        if self.config.is_regression:
            if val_data != None and val_labels != None:
                self.predict(val_data)
                avg_loss = self.output.loss(val_labels) / val_labels.shape[0]
                print 'val_loss %.4f,' % (avg_loss),
            if test_data != None and test_labels != None:
                self.predict(test_data)
                avg_loss = self.output.loss(test_labels) / test_labels.shape[0]
                print 'test_loss %.4f,' % (avg_loss),
        else:
            # print accuracy if it is a classification problem
            ypred = self.predict(train_data)
            acc = (ypred == train_labels.squeeze()).mean()
            print 'acc %.4f,' % acc,

            if val_data != None and val_labels != None:
                ypred = self.predict(val_data)
                acc = (ypred == val_labels.squeeze()).mean()
                print 'val_acc %.4f,' % acc,
            if test_data != None and test_labels != None:
                ypred = self.predict(test_data)
                acc = (ypred == test_labels.squeeze()).mean()
                print 'test_acc %.4f,' % acc,

        if self.config.display_winc:
            for i in range(0, self.num_layers):
                print 'winc%d %.5f,' % (i+1, np.abs(self.layer[i].Winc).max()),
            print 'winc_out %.5f,' % np.abs(self.output.Winc).max(),

        print 'time %.2f' % time

    def _forward(self, X):
        """Do a forward pass without computing the output and predictions.
        Used as a subroutine for function predict and check_grad."""
        Xbelow = X
        for i in range(0, self.num_layers):
            Xbelow = self.layer[i].forward(Xbelow)
        self.output.forward(Xbelow)
       
    def predict(self, X):
        """Make prediction using the current network.
        
        X: N*D data matrix
        ispad: if True, X is padded by an extra dimension of constant 1's

        Return an N-element vector of predicted labels.
        """
        self._forward(X)
        return self.output.predict()

    def read_data(self, data_file_name):
        """(data_file_name) --> data
        Read from the specified data file, return a data object, which is an
        object with three attributes, X, T and K. X and T are the data and
        target matrices respectively, and K is the dimensionality of the output.
        Each of X and T is a matrix with N rows, N is the number of data
        cases."""

        f = open(data_file_name)

        data_dict = pickle.load(f)

        f.close()

        data = Data()
        data.X = data_dict['data']
        data.T = data_dict['labels']
        data.K = data_dict['K']

        return data

    def read_loss(self, loss_file_name):
        """(data_file_name) --> loss
        Read from the specified data file, return a loss matrix.
        """
        f = open(loss_file_name)
        d = pickle.load(f)
        f.close()

        return d['loss']

    def display(self):
        print '%d training cases' % self.train_data.X.shape[0]
        if self.config.is_val:
            print '%d validation cases' % self.val_data.X.shape[0]
        if self.config.is_test:
            print '%d test cases' % self.test_data.X.shape[0]
        print '[' + str(self.output) + ']'
        for i in range(self.num_layers-1, -1, -1):
            print '[' + str(self.layer[i]) + ']'
        print '[input ' + str(self.input_dim) + ']'

        print 'learn_rate : ' + str(self.config.learn_rate)
        print 'init_scale : ' + str(self.config.init_scale)
        print 'momentum : ' + str(self.config.momentum)
        print 'weight_decay : ' + str(self.config.weight_decay)
        print 'minibatch_size : ' + str(self.config.minibatch_size)
        print 'num_epochs : ' + str(self.config.num_epochs)
        print 'epoch_to_save : ' + str(self.config.epoch_to_save)

    def check_grad(self):
        # check the gradient of the 1st layer weights
        import scipy.optimize as opt

        ncases = 100

        def f(w):
            if self.num_layers == 0:
                Wtemp = self.output.W
                self.output.W = w.reshape(Wtemp.shape)
            else:
                Wtemp = self.layer[0].W
                self.layer[0].W = w.reshape(Wtemp.shape)

            self._forward(self.train_data.X[:ncases,:])

            Z = self.train_data.T[:ncases]
            if not self.config.is_regression:
                Z = self.output.act_type.label_vec_to_mat(Z, self.train_data.K)

            L = self.output.loss(Z) / Z.shape[0]
            if self.num_layers == 0:
                self.output.W = Wtemp
            else:
                self.layer[0].W = Wtemp

            return L

        def fgrad(w):
            if self.num_layers == 0:
                Wtemp = self.output.W
                self.output.W = w.reshape(Wtemp.shape)
            else:
                Wtemp = self.layer[0].W
                self.layer[0].W = w.reshape(Wtemp.shape)

            self._forward(self.train_data.X[:ncases,:])

            Z = self.train_data.T[:ncases]
            if not self.config.is_regression:
                Z = self.output.act_type.label_vec_to_mat(Z, self.train_data.K)
            self.output.loss(Z)

            self.output.gradient()
            dLdXabove = self.output.dLdXtop[:,:-1]
            for i in range(self.num_layers-1, -1, -1):
                self.layer[i].gradient(dLdXabove)
                dLdXabove = self.layer[i].dLdXbelow[:,:-1]

            if self.num_layers == 0:
                grad_w = self.output.dLdW
            else:
                grad_w = self.layer[0].dLdW

            if self.num_layers == 0:
                self.output.W = Wtemp
            else:
                self.layer[0].W = Wtemp

            return grad_w.reshape(np.prod(grad_w.shape)) / Z.shape[0]

        if self.num_layers == 0:
            #W = np.random.randn(
            #        self.output.W.shape[0], self.output.W.shape[1]) * 1e-3
            W = self.output.W
        else:
            #W = np.random.randn(
            #        self.layer[0].W.shape[0], self.layer[0].W.shape[1]) * 1e-3
            W = self.layer[0].W

        print "wmax: %f" % np.abs(fgrad(W.reshape(np.prod(W.shape)))).max()
        print "check_grad err: %f" % opt.check_grad(
                f, fgrad, W.reshape(np.prod(W.shape)))


class NNStore:
    """An object containing all parameters of the neural network, made easy to
    store and load networks."""
    def __init__(self):
        pass

    def init_from_net(self, net):
        """net should be an instance of NN."""
        self.num_layers = net.num_layers
        self.layer = []
        for i in range(0, self.num_layers):
            layer = LayerStore()
            layer.W = net.layer[i].W
            layer.act_type = net.layer[i].act_type
            self.layer.append(layer)

        output = LayerStore()
        output.W = net.output.W
        output.act_type = net.output.act_type

        self.output = output

    def update_from_net(self, net):
        """Update the weights at each layer in a net."""
        for i in range(0, self.num_layers):
            self.layer[i].W = net.layer[i].W
        self.output.W = net.output.W

    def write(self, file_name):
        """Write the net to a file."""
        f = open(file_name, mode='w')
        pickle.dump(self, f)
        f.close()

    def load(self, file_name):
        """Load a net from a file."""
        f = open(file_name)
        nnstore = pickle.load(f)
        f.close()

        self.num_layers = nnstore.num_layers
        self.layer = nnstore.layer
        self.output = nnstore.output

        del nnstore

