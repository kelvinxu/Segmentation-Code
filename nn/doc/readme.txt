** Format of the cfg file **

The file contains at least 3 sections, with section names 'data',
'parameters' and 'output'.  It is also possible to have one or more 'layer'
sections, the name of layer sections should be in the format of 'layer%d',
where the number (starting from 1) indexes the hidden layers in the network.
The layer indices should be continuous integers starting from 1.

** Format of data files **

The data files should be pickled dictionaries that contain 3 entries: 'data',
'labels' and 'K'.  'data' is a N*D data matrix, each row is a vector for one
data point.  'labels' can be a N*K matrix (for regression problems) or N-D
vector (for classification problems), depend on the specific problem.  If 
'labels' is a vector or list and the problem is classification, it will be 
transformed into N*K matrix using a 1-of-K representation.  'K' is an inteter,
representing the number of different classes or dimensionality of the output 
in regression problems. The class index should start from 0 for the 1-of-K 
representation transform to work properly.
