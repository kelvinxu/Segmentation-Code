from layer import Layer, OutputLayer, LayerConfig, LayerStore
import numpy as np
import gnumpy as gnp
import cPickle as pickle
import time
import act
import os

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
        self.task_loss_fn = None

    def load_train_data(self, data):
        self.train_data = data
        self.train_data.X = gnp.garray(data.X)

    def load_val_data(self, data):
        self.val_data = data
        self.val_data.X = gnp.garray(data.X)

    def load_test_data(self, data):
        self.test_data = data
        self.test_data.X = gnp.garray(data.X)

    def init_net_without_loading_data(self, config):
        """This should be called after loading all required data."""
        self.config = config

        if config.is_output and (not os.path.exists(config.output_dir)):
            os.makedirs(config.output_dir)

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
        for i in range(self.num_layers):
            layer_spec = config.layer[i]
            self.layer.append(Layer(
                in_dim, layer_spec.out_dim, layer_spec.act_type, 
                layer_spec.weight_decay, layer_spec.weight_constraint, 
                layer_spec.dropout))
            in_dim = layer_spec.out_dim

        self.output = OutputLayer(in_dim, config.output.out_dim,
                config.output.output_type, config.output.weight_decay,
                config.output.weight_constraint, config.output.dropout)

        # if not linear output (regression) load task loss function
        if not isinstance(self.output.act_type, act.LinearOutput):
            if config.task_loss_file != None:
                self.task_loss = self.read_loss(config.task_loss_file)
                print 'Loading task loss from %s' % config.task_loss_file
            else:
                self.task_loss = 1 - np.eye(self.train_data.K)
                print 'No task loss specified, using 0-1 loss.'

        # To use multi-class hinge output, a training loss function is required
        if isinstance(self.output.act_type, act.MulticlassHingeOutput):
            if config.train_loss_file != None:
                self.train_loss = self.read_loss(config.train_loss_file)
                print 'Loading surrogate loss from %s' % config.train_loss_file
            else:
                self.train_loss = 1 - np.eye(self.train_data.K)
                print 'No surrogate loss specified, using 0-1 loss.'
            self.output.act_type.set_loss(self.train_loss)

        # initialize the weights in every layer
        self._init_weights(config.init_scale, config.random_seed)

    def init_net(self, config):
        """config is an instance of class Config"""
        self.train_data = self.read_data(config.train_data_file)
        print 'Loading training data from %s' % config.train_data_file

        if config.is_val:
            self.val_data = self.read_data(config.val_data_file)
            print 'Loading validation data from %s' % config.val_data_file
        if config.is_test:
            self.test_data = self.read_data(config.test_data_file)
            print 'Loading test data from %s' % config.test_data_file

        self.init_net_without_loading_data(config)

    def load_net(self, model_file):
        """Load a saved model from a specified file."""
        nnstore = NNStore()
        nnstore.load(model_file)
        self.build_net_from_copy(nnstore)

    def make_copy(self):
        """
        Make a CPU copy of the net. This copy can be used to recover the net.
        """
        nnstore = NNStore()
        nnstore.init_from_net(self)
        return nnstore

    def build_net_from_copy(self, copy):
        """
        Rebuild the net from a copy made by make_copy.
        """
        nnstore = copy
        self.num_layers = len(nnstore.layer)
        self.layer = []
        for i in range(self.num_layers):
            in_dim, out_dim = nnstore.layer[i].W.shape
            new_layer = Layer(in_dim, out_dim, nnstore.layer[i].act_type)
            new_layer.load_weight(nnstore.layer[i].W, nnstore.layer[i].b)
            self.layer.append(new_layer)

        in_dim, out_dim = nnstore.output.W.shape
        new_layer = OutputLayer(in_dim, out_dim, nnstore.output.act_type)
        new_layer.load_weight(nnstore.output.W, nnstore.output.b)
        self.output = new_layer
        if self.num_layers > 0:
            self.input_dim = self.layer[0].W.shape[0]
        else:
            self.input_dim = self.output.W.shape[0]

    def _init_weights(self, init_scale, random_seed=None):
        if random_seed:
            np.random.seed(random_seed)

        for i in range(0, self.num_layers):
            self.layer[i].init_weight(init_scale)

        self.output.init_weight(init_scale)

    def set_task_loss(self, task_loss_fn):
        """Set the task loss function to be user defined task loss.
        
        task_loss_fn should have a signature like this:
        task_loss_fn(OutputType, Y, Z, A)
        """
        self.task_loss_fn = task_loss_fn

    def _compute_loss(self, X, T, batch_size=1000):
        n_total = X.shape[0]
        n_batches = n_total / batch_size
        loss = 0
        for i in range(n_batches):
            gnp.free_reuse_cache()
            i_start = i * batch_size
            if i < n_batches - 1:
                i_end = i_start + batch_size
            else:
                i_end = n_total

            Xbatch = X[i_start:i_end]
            Tbatch = T[i_start:i_end]

            self._forward(Xbatch)
            loss += self.output.loss(Tbatch)
        
        return loss / n_total

    def train(self):
        config = self.config

        # convert t into a matrix in 1-of-K representation if it is a vector
        t = self.train_data.T
        T_matrix = self.output.act_type.label_vec_to_mat(t, self.train_data.K)

        layer_config = LayerConfig()
        layer_config.learn_rate = config.learn_rate
        layer_config.momentum = config.init_momentum
        layer_config.weight_decay = config.weight_decay

        nnstore = NNStore()
        nnstore.init_from_net(self)

        best_net = NNStore()
        best_net.init_from_net(self)

        train_acc, val_acc, test_acc = self.display_training_info(
                -1, 
                self._compute_loss(
                    self.train_data.X, T_matrix, config.minibatch_size),
                0)
        acc_rec = np.zeros((config.num_epochs / config.epoch_to_display + 1, 4))
        acc_rec[0, 0] = 0
        acc_rec[0, 1] = train_acc
        if config.is_val:
            acc_rec[0, 2] = val_acc
        if config.is_test:
            acc_rec[0, 3] = test_acc

        t_start = time.time()

        best_acc = val_acc
        if self.config.is_test:
            best_test_acc = test_acc
        best_epoch = -1

        for epoch in range(0, config.num_epochs):
            gnp.free_reuse_cache()

            # decrease learning rate over time
            layer_config.learn_rate = config.learn_rate / \
                    (epoch / config.lr_drop_rate + 1)

            # TODO [dirty] special for Lnsvm
            if isinstance(self.output.act_type, act.LnsvmVariantOutput):
                #self.output.act_type.n = 3.0 - (3.0 - 0.5) / 50 * epoch
                self.output.act_type.n = 0.5
                if self.output.act_type.n < 0.5:
                    self.output.act_type.n = 0.5 

                if (epoch + 1) % config.epoch_to_display == 0:
                    print 'n %.4f' % self.output.act_type.n,
            
            if epoch >= config.switch_epoch:
                layer_config.momentum = config.final_momentum

            # shuffle the dataset 
            idx = np.random.permutation(self.num_total_cases)
            #idx = np.arange(self.num_total_cases)
            train_X = self.train_data.X[idx]
            train_T = T_matrix[idx]

            if config.input_noise > 0:
                train_X = train_X * (gnp.rand(train_X.shape) > config.input_noise)
                # train_X = train_X + gnp.randn(train_X.shape) * config.input_noise

            loss = 0

            for batch in range(0, self.num_minibatches):
                i_start = batch * config.minibatch_size
                if not batch == self.num_minibatches - 1:
                    i_end = i_start + config.minibatch_size
                else:
                    i_end = self.num_total_cases

                X = train_X[i_start:i_end]
                T = train_T[i_start:i_end]

                # forward pass
                self._forward(X)

                # compute loss
                loss += self.output.loss(T)

                if self.output.Y.isnan().any():
                    import ipdb
                    ipdb.set_trace()
                    print 'batch #%d <-- nan' % batch

                # backprop
                dLdXabove = self.output.backprop(layer_config)
                for i in range(self.num_layers-1, -1, -1):
                    dLdXabove = self.layer[i].backprop(dLdXabove, layer_config)

            # statistics
            avg_loss = 1.0 * loss / self.num_total_cases

            if (epoch + 1) % config.epoch_to_display == 0:
                train_acc, val_acc, test_acc = self.display_training_info(
                        epoch, avg_loss, time.time() - t_start)

                if val_acc == None:
                    val_acc = train_acc

                if (config.show_task_loss and val_acc < best_acc) or \
                        (not config.show_task_loss and val_acc > best_acc):
                    best_acc = val_acc
                    best_net.update_from_net(self)
                    if config.is_test:
                        best_test_acc = test_acc
                    best_epoch = epoch
                t_start = time.time()
                acc_rec[(epoch + 1) / config.epoch_to_display, 0] = epoch + 1
                acc_rec[(epoch + 1) / config.epoch_to_display, 1] = train_acc
                if config.is_val:
                    acc_rec[(epoch + 1) / config.epoch_to_display, 2] = val_acc
                if config.is_test:
                    acc_rec[(epoch + 1) / config.epoch_to_display, 3] = test_acc

            if (epoch + 1) % config.epoch_to_save == 0:
                nnstore.update_from_net(self)
                nnstore.write(config.output_dir + '/m' + str(epoch + 1) + '.pdata')


        print '----------------------------------------------------------------'

        if config.show_task_loss:
            s = 'loss'
        else:
            s = 'acc'
        
        if config.is_val:
            print 'Best val_%s %.4f' % (s, best_acc),
        else:
            print 'Best train_%s %.4f' % (s, best_acc),

        if config.is_test:
            print '--> test_%s %.4f' % (s, best_test_acc),
        print 'at epoch %d' % (best_epoch + 1)

        if config.is_output:
            f = open('%s/acc_rec.pdata' % config.output_dir, 'w')
            pickle.dump(acc_rec, f, -1)
            f.close()

            self.write_config('%s/cfg.txt' % config.output_dir)

            # save the best net
            fname = config.output_dir + '/best_net.pdata'
            print 'Saving the best model to ' + fname
            best_net.write(fname)

        if config.is_test:
            return (best_acc, best_test_acc)
        else:
            return (best_acc)

    def display_training_info(self, epoch, loss, time):
        """Print training information. Use the config information to determine
        what information to display.

        Return a 3-tuple (train acc, val acc, test acc)
        val acc and test acc will be 0 if no validation/test data are given
        """
        if self.config.is_val:
            if self.config.is_test:
                return self._display_training_info(epoch, loss, time,
                        self.train_data.X, self.train_data.T,
                        val_data=self.val_data.X, val_labels=self.val_data.T,
                        test_data=self.test_data.X, test_labels=self.test_data.T)
            else:
                return self._display_training_info(epoch, loss, time,
                        self.train_data.X, self.train_data.T,
                        val_data=self.val_data.X, val_labels=self.val_data.T)
        else:
            if self.config.is_test:
                return self._display_training_info(epoch, loss, time,
                        self.train_data.X, self.train_data.T,
                        test_data=self.test_data.X, test_labels=self.test_data.T)
            else:
                return self._display_training_info(epoch, loss, time,
                        self.train_data.X, self.train_data.T)

    def _display_training_info(self, epoch, loss, time, 
            train_data, train_labels, val_data=None, val_labels=None, 
            test_data=None, test_labels=None):
        """Print training information during training."""
        print 'epoch %d, surrogate loss %.4f,' % (epoch + 1, loss),

        train_acc = 0
        val_acc = None
        test_acc = None
        acc = 0

        # print loss if it is a regression problem
        if self.config.is_regression:
            # TODO [Dirty code]
            #self.predict(train_data)
            #avg_loss = self.output.task_loss(train_labels, self.task_loss_fn)
            avg_loss = np.sqrt(self._compute_loss(train_data, train_labels) * 2)
            print 'train_loss %.4f,' % avg_loss,

            if val_data != None and val_labels != None:
                #self.predict(val_data)
                #avg_loss = self.output.task_loss(val_labels, self.task_loss_fn)
                avg_loss = np.sqrt(self._compute_loss(val_data, val_labels) * 2)
                print 'val_loss %.4f,' % (avg_loss),
                val_acc = avg_loss
            if test_data != None and test_labels != None:
                #self.predict(test_data)
                #avg_loss = self.output.task_loss(test_labels, self.task_loss_fn)
                avg_loss = np.sqrt(self._compute_loss(test_data, test_labels) * 2)
                print 'test_loss %.4f,' % (avg_loss),
                test_acc = avg_loss
        else:
            # print accuracy if it is a classification problem
            ypred = self.predict(train_data)
            if self.config.show_accuracy:
                acc = (ypred == train_labels.squeeze()).mean()
                print 'acc %.4f,' % acc,
            if self.config.show_task_loss:
                acc = self.task_loss[ypred, train_labels].mean()
                print 'loss %.4f,' % acc,

            train_acc = acc

            if val_data != None and val_labels != None:
                ypred = self.predict(val_data)
                if self.config.show_accuracy:
                    acc = (ypred == val_labels.squeeze()).mean()
                    print 'val_acc %.4f,' % acc,
                if self.config.show_task_loss:
                    acc = self.task_loss[ypred, val_labels].mean()
                    print 'val_loss %.4f,' % acc,
                val_acc = acc
            if test_data != None and test_labels != None:
                ypred = self.predict(test_data)
                if self.config.show_accuracy:
                    acc = (ypred == test_labels.squeeze()).mean()
                    print 'test_acc %.4f,' % acc,
                if self.config.show_task_loss:
                    acc = self.task_loss[ypred, test_labels].mean()
                    print 'test_loss %.4f,' % acc,
                test_acc = acc

        if self.config.display_winc:
            self.display_winc()

        print 'time %.2f' % time

        return (train_acc, val_acc, test_acc)

    def display_winc(self):
        """Display scale of weight updates. This can be used by external
        applications."""
        for i in range(0, self.num_layers):
            print 'winc%d %.5f,' % (i+1, gnp.abs(self.layer[i].Winc).max()),
        print 'winc_out %.5f,' % gnp.abs(self.output.Winc).max(),

    def _forward(self, X):
        """Do a forward pass without computing the output and predictions.
        Used as a subroutine for function predict and check_grad."""
        Xbelow = X
        for i in range(self.num_layers):
            Xbelow = self.layer[i].forward(Xbelow)
        self.output.forward(Xbelow)
       
    def predict(self, X):
        """Make prediction using the current network.
        
        X: N*D data matrix

        Return an N-element vector of predicted labels.
        """
        self._forward(X)
        return self.output.predict()

    def forward(self, X):
        """Compute the activation for each class.
        
        X: N*D data matrix

        Return a N*D activation matrix A.
        """
        self._forward(X)
        return self.output.A

    def _backprop(self, config):
        """Backpropagate through the net from the output layer. This will be
        used as an external interface for semi-supervised application, and the
        backprop starts from the `update_weights` method of the output layer,
        rather than the `backprop` method."""
        dLdXabove = self.output.update_weights(config)
        for i in range(self.num_layers-1, -1, -1):
            dLdXabove = self.layer[i].backprop(dLdXabove, config)

    def eval_task_loss(self, X, z, loss):
        """Evaluate the performance of the net using task specific loss.
        Classification problems only.

        X: N*D data matrix
        z: N-d ground truth matrix.
        loss: K*K matrix, K is the number of classes.

        Return the average loss over all datacases.
        """
        y = self.predict(X)
        return loss[z, y].mean()

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
        data.X = gnp.garray(data_dict['data'])
        #data.T = data_dict['labels'].astype(np.float)
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

        return d

    def write_config(self, filename):
        f = open(filename, 'w')
        f.write('%d training cases\n' % self.train_data.X.shape[0])
        if self.config.is_val:
            f.write('%d validation cases\n' % self.val_data.X.shape[0])
        if self.config.is_test:
            f.write('%d test cases\n' % self.test_data.X.shape[0])
        f.write('[' + str(self.output) + ']\n')
        for i in range(self.num_layers-1, -1, -1):
            f.write('[' + str(self.layer[i]) + ']\n')
        f.write('[input ' + str(self.input_dim) + ']\n')

        f.write('learn_rate : ' + str(self.config.learn_rate) + '\n')
        f.write('init_scale : ' + str(self.config.init_scale) + '\n')
        f.write('init_momentum : ' + str(self.config.init_momentum) + '\n')
        f.write('switch_epoch : ' + str(self.config.switch_epoch) + '\n')
        f.write('final_momentum : ' + str(self.config.final_momentum) + '\n')
        f.write('weight_decay : ' + str(self.config.weight_decay) + '\n')
        f.write('minibatch_size : ' + str(self.config.minibatch_size) + '\n')
        f.write('num_epochs : ' + str(self.config.num_epochs) + '\n')
        f.write('epoch_to_save : ' + str(self.config.epoch_to_save) + '\n')

        f.close()

    def display_structure(self):
        print '[' + str(self.output) + ']'
        for i in range(self.num_layers-1, -1, -1):
            print '[' + str(self.layer[i]) + ']'
        print '[input ' + str(self.input_dim) + ']'

    def display(self):
        print '%d training cases' % self.train_data.X.shape[0]
        if self.config.is_val:
            print '%d validation cases' % self.val_data.X.shape[0]
        if self.config.is_test:
            print '%d test cases' % self.test_data.X.shape[0]

        self.display_structure()

        print 'learn_rate : ' + str(self.config.learn_rate)
        print 'init_scale : ' + str(self.config.init_scale)
        print 'init_momentum : ' + str(self.config.init_momentum)
        print 'switch_epoch : ' + str(self.config.switch_epoch)
        print 'final_momentum : ' + str(self.config.final_momentum)
        print 'weight_decay : ' + str(self.config.weight_decay)
        print 'minibatch_size : ' + str(self.config.minibatch_size)
        print 'num_epochs : ' + str(self.config.num_epochs)
        print 'epoch_to_save : ' + str(self.config.epoch_to_save)

        if self.config.is_output:
            print 'output_dir : ' + self.config.output_dir

    def check_grad(self):
        # check the gradient of the 1st layer weights
        import scipy.optimize as opt

        ncases = 100

        def f(w):
            if self.num_layers == 0:
                Wtemp = self.output.W
                self.output.W = gnp.garray(w.reshape(Wtemp.shape))
            else:
                Wtemp = self.layer[0].W
                self.layer[0].W = gnp.garray(w.reshape(Wtemp.shape))

            self._forward(self.train_data.X[:ncases,:])

            Z = self.train_data.T[:ncases]
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
                self.output.W = gnp.garray(w.reshape(Wtemp.shape))
            else:
                Wtemp = self.layer[0].W
                self.layer[0].W = gnp.garray(w.reshape(Wtemp.shape))

            self._forward(self.train_data.X[:ncases,:])

            Z = self.train_data.T[:ncases]
            Z = self.output.act_type.label_vec_to_mat(Z, self.train_data.K)
            self.output.loss(Z)

            self.output.gradient()
            dLdXabove = self.output.dLdXtop
            for i in range(self.num_layers-1, -1, -1):
                self.layer[i].gradient(dLdXabove)
                dLdXabove = self.layer[i].dLdXbelow

            if self.num_layers == 0:
                grad_w = self.output.dLdW
            else:
                grad_w = self.layer[0].dLdW

            if self.num_layers == 0:
                self.output.W = Wtemp
            else:
                self.layer[0].W = Wtemp

            return grad_w.reshape(np.prod(grad_w.shape)).asarray() / Z.shape[0]

        if self.num_layers == 0:
            W = self.output.W
        else:
            W = self.layer[0].W
        W = W.asarray()

        def finite_diff_grad(f, x0):
            eps = 1e-8
            approx = np.zeros(len(x0))
            for i in xrange(len(x0)):
                x0plus = x0.copy()
                x0minus = x0.copy()
                x0plus[i] += eps
                x0minus[i] -= eps
                approx[i] = (f(x0plus) - f(x0minus)) / (2 * eps)
            return approx

        net_grad = fgrad(W.reshape(W.size))
        fd_grad = finite_diff_grad(f, W.reshape(W.size))
        print "wmax: %f" % np.abs(net_grad).max()
        print "finite difference grad scale: %f" % np.abs(fd_grad).max()
        print "check_grad err: %f" % np.sqrt(((fd_grad - net_grad)**2).sum())

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
            layer.W = net.layer[i].W.asarray()
            layer.b = net.layer[i].b.asarray()
            layer.act_type = net.layer[i].act_type
            self.layer.append(layer)

        output = LayerStore()
        output.W = net.output.W.asarray()
        output.b = net.output.b.asarray()
        output.act_type = net.output.act_type

        self.output = output

    def update_from_net(self, net):
        """Update the weights at each layer in a net."""
        for i in range(0, self.num_layers):
            self.layer[i].W = net.layer[i].W.asarray()
            self.layer[i].b = net.layer[i].b.asarray()
        self.output.W = net.output.W.asarray()
        self.output.b = net.output.b.asarray()

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

