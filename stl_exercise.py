#coding=utf-8
#使用稀疏编码的方法获取特征，然后用softmax来分类并调整中间参数
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from numpy import *


#计算softmax代价
def softmax_cost(theta, n_classes, input_size, lambda_, data, labels):
    k = n_classes
    n, m = data.shape
    theta = theta.reshape((k, n))
    theta_data = theta.dot(data)
    alpha = np.max(theta_data, axis=0)
    theta_data -= alpha 
    proba = np.exp(theta_data) / np.sum(np.exp(theta_data), axis=0)
    indicator = scipy.sparse.csr_matrix((np.ones(m), (labels, np.arange(m))))
    indicator = np.array(indicator.todense())
    cost = -1.0/m * np.sum(indicator * np.log(proba)) + 0.5*lambda_*np.sum(theta*theta)
    grad = -1.0/m * (indicator - proba).dot(data.T) + lambda_*theta
    grad = grad.ravel()

    return cost, grad


#自动编码前向传播
def feedforward_autoencoder(theta, hidden_size, visible_size, data):

    W1 = theta[0 : hidden_size*visible_size].reshape((hidden_size, visible_size))
    b1 = theta[2*hidden_size*visible_size : 2*hidden_size*visible_size+hidden_size].reshape((-1, 1))
    
    a1 = data
    z2 = W1.dot(a1) + b1
    a2 = sigmoid(z2)

    return a2


#sigmod函数
def sigmoid(inX):
    return 1.0/(1+exp(-inX)) 


def sigmoid_prime(x):
    f = sigmoid(x)
    df = f*(1.0-f)
    return df

#参数初始化
def initialize_parameters(hidden_size, visible_size):
    r  = np.sqrt(6) / np.sqrt(hidden_size + visible_size + 1)
    # we'll choose weights uniformly from the interval [-r, r)
    W1 = np.random.random((hidden_size, visible_size)) * 2.0 * r - r
    W2 = np.random.random((visible_size, hidden_size)) * 2.0 * r - r

    b1 = np.zeros(hidden_size)
    b2 = np.zeros(visible_size)
    theta = np.hstack((W1.ravel(), W2.ravel(), b1.ravel(), b2.ravel()))

    return theta


def KL_divergence(p, q):
    return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))



#稀疏编码代价计算
def sparse_autoencoder_cost(theta, visible_size, hidden_size, lambda_, sparsity_param, beta, data):

    W1 = theta[0 : hidden_size*visible_size].reshape((hidden_size, visible_size))
    W2 = theta[hidden_size*visible_size : 2*hidden_size*visible_size].reshape((visible_size, hidden_size))
    b1 = theta[2*hidden_size*visible_size : 2*hidden_size*visible_size+hidden_size]
    b2 = theta[2*hidden_size*visible_size+hidden_size:]

    # Number of instances
    m = data.shape[1]

    # Forward pass
    a1 = data              # Input activation
    z2 = W1.dot(a1) + b1.reshape((-1, 1))
    a2 = sigmoid(z2)
    z3 = W2.dot(a2) + b2.reshape((-1, 1))
    h  = sigmoid(z3)       # Output activation
    y  = a1

    # Compute rho_hat used in sparsity penalty
    rho = sparsity_param
    rho_hat = np.mean(a2, axis=1)
    sparsity_delta = (-rho/rho_hat + (1.0-rho)/(1-rho_hat)).reshape((-1, 1))

    # Backpropagation
    delta3 = (h-y)*sigmoid_prime(z3)
    delta2 = (W2.T.dot(delta3) + beta*sparsity_delta)*sigmoid_prime(z2)

    # Compute the cost
    squared_error_term = np.sum((h-y)**2) / (2.0*m)
    weight_decay = 0.5*lambda_*(np.sum(W1*W1) + np.sum(W2*W2))
    sparsity_term = beta*np.sum(KL_divergence(rho, rho_hat))
    cost = squared_error_term + weight_decay + sparsity_term

    # Compute the gradients
    W1grad = delta2.dot(a1.T)/m + lambda_*W1
    W2grad = delta3.dot(a2.T)/m + lambda_*W2
    b1grad = np.mean(delta2, axis=1)
    b2grad = np.mean(delta3, axis=1)
    grad = np.hstack((W1grad.ravel(), W2grad.ravel(), b1grad, b2grad))

    return cost, grad


#参数可视化
def display_network(A):
    opt_normalize = True
    opt_graycolor = True

    # Rescale
    A = A - np.average(A)

    # Compute rows & cols
    (row, col) = A.shape
    sz = int(np.ceil(np.sqrt(row)))
    buf = 1
    n = np.ceil(np.sqrt(col))
    m = np.ceil(col / n)

    image = np.ones(shape=(buf + m * (sz + buf), buf + n * (sz + buf)))

    if not opt_graycolor:
        image *= 0.1

    k = 0
    for i in range(int(m)):
        for j in range(int(n)):
            if k >= col:
                continue

            clim = np.max(np.abs(A[:, k]))

            if opt_normalize:
                image[buf + i * (sz + buf):buf + i * (sz + buf) + sz, buf + j * (sz + buf):buf + j * (sz + buf) + sz] = \
                    A[:, k].reshape(sz, sz) / clim
            else:
                image[buf + i * (sz + buf):buf + i * (sz + buf) + sz, buf + j * (sz + buf):buf + j * (sz + buf) + sz] = \
                    A[:, k].reshape(sz, sz) / np.max(np.abs(A))
            k += 1

    return image


#softmax训练
def softmax_train(input_size, n_classes, lambda_, input_data, labels, options={'maxiter': 400, 'disp': True}):
    theta = 0.005 * np.random.randn(n_classes * input_size)

    J = lambda theta : softmax_cost(theta, n_classes, input_size, lambda_, input_data, labels)

    # Find out the optimal theta
    results = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options)
    opt_theta = results['x']

    model = {'opt_theta': opt_theta, 'n_classes': n_classes, 'input_size': input_size}
    return model


def softmax_predict(model, data):
    
    theta = model['opt_theta'] # Optimal theta
    k = model['n_classes']  # Number of classes
    n = model['input_size'] # Input size (number of features)

    # Reshape theta
    theta = theta.reshape((k, n))

    # Probability with shape (k, m)
    theta_data = theta.dot(data)
    alpha = np.max(theta_data, axis=0)
    theta_data -= alpha # Avoid numerical problem due to large values of exp(theta_data)
    proba = np.exp(theta_data) / np.sum(np.exp(theta_data), axis=0)

    # Prediction values
    pred = np.argmax(proba, axis=0)

    return pred

def load_MNIST_images(filename):	#加载
    with open(filename, "r") as f:
        magic = np.fromfile(f, dtype=np.dtype('>i4'), count=1)

        n_images = np.fromfile(f, dtype=np.dtype('>i4'), count=1)
        rows = np.fromfile(f, dtype=np.dtype('>i4'), count=1)
        cols = np.fromfile(f, dtype=np.dtype('>i4'), count=1)

        images = np.fromfile(f, dtype=np.ubyte)
        images = images.reshape((n_images, rows * cols))
        images = images.T
        images = images.astype(np.float64) / 255
        f.close()
        return images


def load_MNIST_labels(filename):	#label加载
    with open(filename, 'r') as f:
        magic = np.fromfile(f, dtype=np.dtype('>i4'), count=1)
        n_labels = np.fromfile(f, dtype=np.dtype('>i4'), count=1)
        labels = np.fromfile(f, dtype=np.uint8)
        f.close()
        return labels


def train_stl_spare(hidden_size, input_size,lambda_, sparsity_param, beta, unlabeled_data):
	#稀疏编码器计算
	theta = initialize_parameters(hidden_size, input_size)
	J = lambda theta : sparse_autoencoder_cost(theta, input_size, hidden_size,
	    lambda_, sparsity_param, beta, unlabeled_data)

	options = {'maxiter': maxiter, 'disp': True}
	results = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options)
	opt_theta = results['x']
	np.save("stl_spare.npy",opt_theta)
	print("Show the results of optimization as following.\n")
	print(results)

	# Visualize weights
	W1 = opt_theta[0:hidden_size*input_size].reshape((hidden_size, input_size))
	image = display_network(W1.T)
	plt.figure()
	plt.imsave('stl_weights.png', image, cmap=plt.cm.gray)
	plt.imshow(image, cmap=plt.cm.gray)

input_size  = 28 * 28
n_labels  = 5
hidden_size = 200
sparsity_param = 0.1 # desired average activation of the hidden units.
                     # (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
                     #  in the lecture notes).
lambda_ = 3e-3       # weight decay parameter
beta = 3             # weight of sparsity penalty term
maxiter = 400



# 数据加载
mnist_data   = load_MNIST_images('data/mnist/train-images-idx3-ubyte')
mnist_labels = load_MNIST_labels('data/mnist/train-labels-idx1-ubyte')

labeled_set   = np.argwhere(mnist_labels < 5).flatten()
unlabeled_set = np.argwhere(mnist_labels >= 5).flatten()

n_train = round(labeled_set.size / 2) # Number of training data
train_set = labeled_set[:n_train]
test_set  = labeled_set[n_train:]

train_data   = mnist_data[:, train_set]
train_labels = mnist_labels[train_set]

test_data   = mnist_data[:, test_set]
test_labels = mnist_labels[test_set]

unlabeled_data = mnist_data[:, unlabeled_set]

print('# examples in unlabeled set: {}'.format(unlabeled_data.shape[1]))
print('# examples in supervised training set: {}'.format(train_data.shape[1]))
print('# examples in supervised testing set: {}\n'.format(test_data.shape[1]))


#训练,只需要训练一次
#train_stl_spare(hidden_size, input_size,lambda_, sparsity_param, beta, unlabeled_data)

#获取特征模型
opt_theta=np.load("stl_spare.npy")



np.save("stl_train_features.npy",feedforward_autoencoder(opt_theta, hidden_size, input_size, train_data))
np.save("stl_test_features.npy",feedforward_autoencoder(opt_theta, hidden_size, input_size, test_data))
train_features = np.load("stl_train_features.npy")
test_features  = np.load("stl_test_features.npy")



#SOFTMAX分类器训练
lambda_ = 1e-4 # weight decay parameter
options = {'maxiter': maxiter, 'disp': True}
softmax_model = softmax_train(hidden_size, n_labels, lambda_, train_features, train_labels, options)
print "input_size:",softmax_model['input_size'],"**************\n"
pred = softmax_predict(softmax_model, test_features)

#测试准确率为98%
acc = np.mean(test_labels == pred)
print("The Accuracy (with learned features): {:5.2f}% \n".format(acc*100))
