

def A(x):
    return x**2

def B(x):
    return -x

def C(x):
    return x*3

func_list = { 0:A, 1:B, 2:C }


def test_nn():
    import nn
    import config as cfg

    config = cfg.Config('../784.cfg')
    
    net = nn.NN()
    net.init_net(config)
    net.display()
    net.train()

def prepare_data():
    import cPickle as pickle
    import numpy as np

    cases_per_class = 10
    
    f = open('mnist_train.dat')
    data = pickle.load(f)
    f.close()

    x = data['data']
    t = data['labels']

    xx = np.zeros((cases_per_class * 10, x.shape[1]))
    tt = np.zeros((cases_per_class * 10, 1))

    for i in range(10):
        z = x[(t == i).squeeze()]
        xx[cases_per_class*i:cases_per_class*(i+1),:] = z[:cases_per_class,:]
        tt[cases_per_class*i:cases_per_class*(i+1)] = i

    d = { 'data' : xx, 'labels' : tt, 'K' : 10 }
    f = open('debug_test.pdata', 'w')
    pickle.dump(d, f, -1)
    f.close()

def prepare_data2():
    import cPickle as pickle
    import numpy as np
    import scipy.io as sio

    cases_per_class = 10
    
    f = open('mnist_train.dat')
    data = pickle.load(f)
    f.close()

    x = data['data']
    t = data['labels']

    idx = np.random.permutation(t.size)

    xx = x[idx[:50000],:].astype(np.float) / 255
    tt = t[idx[:50000]].squeeze()

    d = { 'data' : xx, 'labels' : tt, 'K' : 10 }
    f = open('debug_train.pdata', 'w')
    pickle.dump(d, f, -1)
    f.close()
    sio.savemat('debug_train.mat', d)

    xx = x[idx[50000:],:].astype(np.float) / 255
    tt = t[idx[50000:]].squeeze()

    d = { 'data' : xx, 'labels' : tt, 'K' : 10 }
    f = open('debug_val.pdata', 'w')
    pickle.dump(d, f, -1)
    f.close()
    sio.savemat('debug_val.mat', d)

    f = open('mnist_test.dat')
    data = pickle.load(f)
    f.close()

    x = data['data'].astype(np.float) / 255
    t = data['labels'].squeeze()

    d = { 'data' : x, 'labels' : t, 'K' : 10 }
    f = open('debug_test.pdata', 'w')
    pickle.dump(d, f, -1)
    f.close()
    sio.savemat('debug_test.mat', d)

def prepare_binary_data():
    import cPickle as pickle
    import numpy as np

    cases_per_class = 10
    
    f = open('debug_train.pdata')
    data = pickle.load(f)
    f.close()

    x = data['data']
    t = data['labels']

    idx = np.logical_or(t==0, t==1)
    xx = x[idx]
    tt = t[idx]

    d = { 'data' : xx, 'labels' : tt, 'K' : 10 }
    f = open('binary_train.pdata', 'w')
    pickle.dump(d, f, -1)
    f.close()

    f = open('debug_val.pdata')
    data = pickle.load(f)
    f.close()

    x = data['data']
    t = data['labels']

    idx = np.logical_or(t==0, t==1)
    xx = x[idx]
    tt = t[idx]

    d = { 'data' : xx, 'labels' : tt, 'K' : 10 }
    f = open('binary_val.pdata', 'w')
    pickle.dump(d, f, -1)
    f.close()

    f = open('debug_test.pdata')
    data = pickle.load(f)
    f.close()

    x = data['data']
    t = data['labels']

    idx = np.logical_or(t==0, t==1)
    xx = x[idx]
    tt = t[idx]

    d = { 'data' : xx, 'labels' : tt, 'K' : 10 }
    f = open('binary_test.pdata', 'w')
    pickle.dump(d, f, -1)
    f.close()

def f(x):
    return x*x

def fgrad(x):
    return 2*x

def test_checkgrad():
    from scipy.optimize import check_grad
    import numpy as np

    for x in range(100):
        x = x * np.ones((1)) / 10
        print "check_grad @ %.2f: %.6f" % (x, check_grad(f, fgrad, x))

def test_nn_checkgrad():
    import nn
    import config as cfg

    config = cfg.Config('../784.cfg')

    config.num_epochs = 1
    
    net = nn.NN()
    net.init_net(config)
    net.display()
    net.train()
    net.check_grad()

if __name__ == "__main__":
    test_nn()
    # test_checkgrad()
    # test_act()
    # test_nn_checkgrad()
    # print func_list[2](5)
    # prepare_data()
    # prepare_data2()
    # prepare_binary_data()

