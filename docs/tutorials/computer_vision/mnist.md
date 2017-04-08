
# Handwritten Digit Recognition

This tutorial guides you through a classic computer vision application: identify hand written digits with neural networks. 

## Load data

We first fetch the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset, which is a commonly used dataset for handwritten digit recognition. Each image in this dataset has been resized into 28x28 with grayscale value between 0 and 254. The following codes download and load the images and the according labels into `numpy`.




```python
import mxnet as mx
import numpy as np
import os
import wget
import gzip
import struct
import sys

# Choose Hardware CPU or GPU
context = mx.cpu(0)
#context = mx.gpu(0)
```


```python

def download_data(url, force_download=True): 
    fname = url.split("/")[-1]
    if force_download or not os.path.exists(fname):
        wget.download(url, fname)
    return fname

def read_data(label_url, image_url):
    with gzip.open(download_data(label_url)) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    with gzip.open(download_data(image_url), 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
    return (label, image)

path='http://yann.lecun.com/exdb/mnist/'
(train_lbl, train_img) = read_data(
    path+'train-labels-idx1-ubyte.gz', path+'train-images-idx3-ubyte.gz')
(val_lbl, val_img) = read_data(
    path+'t10k-labels-idx1-ubyte.gz', path+'t10k-images-idx3-ubyte.gz')
```

We plot the first 10 images and print their labels. 


```python
%matplotlib inline
import matplotlib.pyplot as plt
import logging
root_logger = logging.getLogger().setLevel(logging.DEBUG)

for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(train_img[i], cmap='Greys_r')
    plt.axis('off')
plt.show()
print('label: %s' % (train_lbl[0:10],))
```

Next we create data iterators for MXNet. The data iterator, which is similar the iterator, returns a batch of data in each `next()` call. A batch contains several images with its according labels. These images are stored in a 4-D matrix with shape `(batch_size, num_channels, width, height)`. For the MNIST dataset, there is only one color channel, and both width and height are 28. In addition, we often shuffle the images used for training, which accelerates the training progress.


```python

def to4d(img):
    return img.reshape(img.shape[0], 1, 28, 28).astype(np.float32)/255

batch_size = 100
train_iter = mx.io.NDArrayIter(to4d(train_img), train_lbl, batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(to4d(val_img), val_lbl, batch_size)

 
```

## Multilayer Perceptron

A multilayer perceptron contains several fully-connected layers. A fully-connected layer, with an *n x m* input matrix *X* outputs a matrix *Y* with size *n x k*, where *k* is often called as the hidden size. This layer has two parameters, the *m x k* weight matrix *W* and the *m x 1* bias vector *b*. It compute the outputs with

$$Y = W X + b.$$

The output of a fully-connected layer is often feed into an activation layer, which performs element-wise operations. Two common options are the sigmoid function, or the rectifier (or "relu") function, which outputs the max of 0 and the input.

The last fully-connected layer often has the hidden size equals to the number of classes in the dataset. Then we stack a softmax layer, which map the input into a probability score. Again assume the input *X* has size *n x m*:

$$ \left[\frac{\exp(x_{i1})}{\sum_{j=1}^m \exp(x_{ij})},\ldots, \frac{\exp(x_{im})}{\sum_{j=1}^m \exp(x_{ij})}\right] $$

Defining the multilayer perceptron in MXNet is straightforward, which has shown as following.


```python
# Create a place holder variable for the input data
data = mx.sym.Variable('data')
# Flatten the data from 4-D shape (batch_size, num_channel, width, height) 
# into 2-D (batch_size, num_channel*width*height)
data = mx.sym.Flatten(data=data)

# The first fully-connected layer
fc1  = mx.sym.FullyConnected(data=data, name='fc1', num_hidden=128)
# Apply relu to the output of the first fully-connnected layer
act1 = mx.sym.Activation(data=fc1, name='relu1', act_type="relu")

# The second fully-connected layer and the according activation function
fc2  = mx.sym.FullyConnected(data=act1, name='fc2', num_hidden = 64)
act2 = mx.sym.Activation(data=fc2, name='relu2', act_type="relu")

# The thrid fully-connected layer, note that the hidden size should be 10, which is the number of unique digits
fc3  = mx.sym.FullyConnected(data=act2, name='fc3', num_hidden=10)
# The softmax and loss layer
mlp  = mx.sym.SoftmaxOutput(data=fc3, name='softmax')

# We visualize the network structure with output size (the batch_size is ignored.)
shape = {"data" : (batch_size, 1, 28, 28)}

```


```python
mx.viz.plot_network(symbol=mlp, shape=shape)
```

Now both the network definition and data iterators are ready. We can start training. 


```python
# @@@ AUTOTEST_OUTPUT_IGNORED_CELL
model = mx.mod.Module(symbol=mlp, 
                    context=context,
                    data_names=['data'], 
                    label_names=['softmax_label'])

model.fit(train_iter, 
        eval_data=val_iter,
        optimizer_params={'learning_rate':0.1},
        num_epoch=10,
        batch_end_callback = mx.callback.Speedometer(batch_size, 200) # output progress for each 200 data batches
         )
```

After training is done, we can predict a single image. 


```python
index_img = 0
plt.imshow(val_img[index_img].reshape((28,28)), cmap='Greys_r')
plt.show()
test_iter = mx.io.NDArrayIter(to4d(val_img), val_lbl, batch_size=100)
prediction_prob = model.predict(eval_data=test_iter,num_batch=1)
clss_pred = prediction_prob[index_img].asnumpy()
print('Result: {}'.format(clss_pred.argmax()),)
```

We can also evaluate the accuracy given a data iterator. 


```python
# @@@ AUTOTEST_OUTPUT_IGNORED_CELL
valid_acc = list(model.score(val_iter,eval_metric=['acc'],num_batch=1,)) #support for python 3 zip
valid_acc = list(model.score(val_iter,eval_metric='acc'))
print ('Validation accuracy: %f%%' % (valid_acc[0][1] *100,))
assert valid_acc[0][1]  > 0.95, "Low validation accuracy."
```

Even more, we can recognizes the digit written on the below box. 


```python
from IPython.display import HTML
import skimage
import numpy as np

def classify(img):
    img = img[len('data:image/png;base64,'):].decode('base64')
    img = np.fromstring(img, np.uint8)
    img = skimage.transform.resize(img[:,:,3], (28,28))
    img = img.astype(np.float32).reshape((1, 784))/255.0
    return model.predict(img)[0].argmax()

'''
To see the model in action, run the demo notebook at
https://github.com/dmlc/mxnet-notebooks/blob/master/python/tutorials/mnist.ipynb.
'''
HTML(filename="mnist_demo.html")
```

## Convolutional Neural Networks

Note that the previous fully-connected layer simply reshapes the image into a vector during training. It ignores the spatial information that pixels are correlated on both horizontal and vertical dimensions. The convolutional layer aims to improve this drawback by using a more structural weight $W$. Instead of simply matrix-matrix multiplication, it uses 2-D convolution to obtain the output. 

<img src="https://thatindiandude.github.io/images/conv.png" style="height: 75%; width: 75%;">

We can also have multiple feature maps, each with their own weight matrices, to capture different features: 
<img src="https://thatindiandude.github.io/images/filters.png" style="height: 75%; width: 75%;">

Besides the convolutional layer, another major change of the convolutional neural network is the adding of pooling layers. A pooling layer reduce a $n\times m$ (often called kernal size) image patch into a single value to make the network less sensitive to the spatial location.

<img src="https://thatindiandude.github.io/images/pooling.png" style="height: 75%; width: 75%;">


```python
data = mx.symbol.Variable('data')
# first conv layer
conv1 = mx.sym.Convolution(data=data, kernel=(5,5), num_filter=20)
tanh1 = mx.sym.Activation(data=conv1, act_type="tanh")
pool1 = mx.sym.Pooling(data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))
# second conv layer
conv2 = mx.sym.Convolution(data=pool1, kernel=(5,5), num_filter=50)
tanh2 = mx.sym.Activation(data=conv2, act_type="tanh")
pool2 = mx.sym.Pooling(data=tanh2, pool_type="max", kernel=(2,2), stride=(2,2))
# first fullc layer
flatten = mx.sym.Flatten(data=pool2)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
tanh3 = mx.sym.Activation(data=fc1, act_type="tanh")
# second fullc
fc2 = mx.sym.FullyConnected(data=tanh3, num_hidden=10)
# softmax loss
lenet = mx.sym.SoftmaxOutput(data=fc2, name='softmax')

```


```python
mx.viz.plot_network(symbol=lenet, shape=shape)
```

Note that LeNet is more complex than the previous multilayer perceptron, so we use GPU instead of CPU for training. 


```python
# @@@ AUTOTEST_OUTPUT_IGNORED_CELL
train_val_log_file = 'mnist_train_val.log'
stream_logger = logging.getLogger()
stream_logger.addHandler(logging.StreamHandler())
stream_logger.addHandler(logging.FileHandler(train_val_log_file, mode='w'))


model = mx.mod.Module(symbol=lenet, 
                    context=context,
                    data_names=['data'], 
                    label_names=['softmax_label'])

model.fit(train_iter,
        eval_data=test_iter,
        optimizer_params={'learning_rate':0.1},
        num_epoch=10,batch_end_callback= mx.callback.Speedometer(batch_size, 200))

score=list(model.score(val_iter,eval_metric=['acc']) )
acc = score[0][1]
assert acc > 0.98, "Low validation accuracy."
```


```python
def parse_log(log_file,metric='accuracy'):
    TR_RE = re.compile('.*?]\sTrain-'+metric+'=([\d\.]+)')
    VA_RE = re.compile('.*?]\sValidation-'+metric+'=([\d\.]+)')
    log = open(log_file).read()
    train_metric = [float(x) for x in TR_RE.findall(log)]
    validation_metric = [float(x) for x in VA_RE.findall(log)]
    return (train_metric,validation_metric)

def plot_curves(train_metric,validation_metric,metric='accuracy',ylim=[0,1]):
    idx = np.arange(len(train_metric))
    plt.figure(figsize=(8, 6))
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.plot(idx, train_metric, 'o', linestyle='-', color="r",
             label="Train "+metric)

    plt.plot(idx, validation_metric, 'o', linestyle='-', color="b",
             label="Validation "+metric)

    plt.legend(loc="best")
    plt.xticks(np.arange(min(idx), max(idx)+1, 5))
    plt.yticks(np.arange(ylim[0], ylim[1], 0.2))
    plt.ylim(ylim)
    plt.show()
```


```python
train_metric,validation_metric=parse_log(train_val_log_file)
plot_curves(train_metric,validation_metric)
```

Note that, with the same hyper-parameters, LeNet achieves 98.7% validation accuracy, which improves on the previous multilayer perceptron accuracy of 96.6%.

Because we rewrite the model parameters in `mod`, now we can try the previous digit recognition box again to check if or not the new CNN model improves the classification accuracy
