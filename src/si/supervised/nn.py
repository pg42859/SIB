import numpy as np
from src.si.util.activation import *
from src.si.util.metrics import mse, mse_prime
from src.si.supervised.Model import Model
from src.si.util.im2col import pad2D, im2col, col2im


class Layer(ABC):
    def __init__(self):
        self.input = None
        self.output = None

    @abstractmethod
    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_error, lr):
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, inputsize, outputsize):
        self.weights = np.random.rand(inputsize, outputsize) - 0.5
        self.bias = np.zeros((1, outputsize))

    def setWeights(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def forward(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, output_error, learning_rate):
        # dE/dW ? X.T * dE/dY
        weights_error = np.dot(self.input.T, output_error)
        # dE/dB = dE/dY
        bias_error = np.sum(output_error, axis=0)
        # dE/dX
        input_error = np.dot(output_error, self.weights.T)
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * bias_error
        return input_error


class Activation(Layer):
    def __init__(self, activation):
        self.function = activation

    def forward(self,input):
        self.input = input
        self.output = self.function(input)
        return self.output

    def backward(self,output_error, lr):
        return np.multiply(self.function.prime(self.input),output_error)


class NN(Model):
    def __init__(self, epochs=1000, lr=0.001, verbose=True):
        self.epochs = epochs
        self.lr = lr
        self.verbose = verbose
        self.layers = []
        self.loss = mse
        self.loss_prime = mse_prime

    def add(self, layer):
        self.layers.append(layer)

    def fit(self, dataset):
        X, Y = dataset.getXy()
        self.dataset = dataset
        self.history = dict()
        for epoch in range(self.epochs):
            output = X
            #forward propagation
            for layer in self.layers:
                output = layer.forward(output)

            # backward propagation
            error = self.loss_prime(Y, output)
            for layer in reversed(self.layers):
                error = layer.backward(error, self.lr)

            err = self.loss(Y, output)
            self.history[epoch] = err
            if self.verbose:
                print(f"epoch {epoch+1}/{self.epochs} error={err}")
            else:
                print("\r", f"epoch {epoch + 1}/{self.epochs} error = {err}")
        self.is_fited = True

    def predict(self,input_data):
        assert self.is_fited
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def cost(self, X=None, y=None):
        assert self.is_fited, 'Model must be fitted'
        X = X if X is not None else self.dataset.X
        y = y if y is not None else self.dataset.y
        output = self.predict(X)
        return self.loss(y, output)


class Flatten(Layer):
    def forward(self, input):
        self.input_shape= input.shape
        output = input.reshape(input.shape[0], -1)
        return output

    def backward(self, output_error, lr):
        return output_error.reshape(self.input_shape)


class Conv2D(Layer):
    def __init__(self, input_shape,kernel_shape, layer_depth, stride = 1, padding = 0):
        self.input_shape = input_shape
        self.in_ch = input_shape[2]
        self.out_ch = layer_depth
        self.stride = stride
        self.padding = padding

        self.weights = np.random.rand(kernel_shape[0],kernel_shape[1],
                                      self.in_ch, self.out_ch) -0.5

        self.bias = np.zeros((self.out_ch,1))

    def forward(self,input_data):
        s = self.stride
        self.X_shape = input_data.shape
        _, p = pad2D(input_data, self.padding, self.weights.shape[:2], s)

        pr1, pr2, pc1, pc2 = p
        fr, fc, in_ch, out_ch = self.weights.shape
        n_ex, in_rows, in_cols, in_ch = input_data.shape

        # compute the dimensions of the convolution output
        out_rows = int((in_rows + pr1 + pr2-fr) / s + 1)
        out_cols = int((in_cols + pc1 + pc2 -fc) / s + 1)

        # convert X and w into the appropriate 2D matrices and take their product
        self.X_col, _ = im2col(input_data, self.weights.shape, p, s)
        W_col = self.weights.transpose(3, 2, 0, 1).reshape(out_ch, -1)

        output_data = (W_col @ self.X_col + self.bias).reshape(out_ch, out_rows, out_cols, n_ex).transpose(3, 1, 2, 0)
        return output_data

    def backward(self, output_error, learning_rate):
        fr, fc, in_ch, out_ch = self.weights.shape
        p = self.padding

        db = np.sum(output_error, axis=(0, 1, 2))
        db = db.reshape(out_ch,)

        dout_reshaped = output_error.transpose(1, 2, 3, 0).reshape(out_ch, -1)
        dW = dout_reshaped @ self.X_col.T
        dW = dW.reshape(self.weights.shape)

        W_reshape = self.weights.reshape(out_ch, -1)
        dx_col = W_reshape. T @ dout_reshaped
        input_error = col2im(dx_col, self.X_shape, self.weights.shape, (p, p, p, p), self.stride)

        self.weights -= learning_rate*dW
        self.bias -= learning_rate*db

        return input_error

# fazer pol


class Poling2D(Model):
    def __init__(self, size, stride):
        self.size = size
        self.stride = stride

    def pool(self, X_col):
        raise NotImplementedError

    def dpool(self, dX_col, size, stride, padding, max_idx):
        raise NotImplementedError

    def forward(self, input):
        self.X_shape = input.shape
        n, h, w, d = input.shape  # comprimento, altura e numero das imagens
        h_out = (h - self.size) / self.stride + 1
        w_out = (w - self.size) / self.stride + 1
        if not w_out.is_intiger() or not h_out.is_intiger():
            raise Exception("Invalid output dimension")
        h_out, w_out = int(h_out), int(w_out)
        X_reshaped = input.reshape()
        self.X_col = im2col(X_reshaped, self.size, pad=0, stride=self.stride)

        out, self.max_idx = self.pool(self.X_col)
        out = out.reshape(h_out, w_out, n, d)
        out = out.transpose(3, 2, 0, 1)
        return out

    def backward(self, output_error, learning_rate):
        n, w, h, d = self.X_shape
        dX_col = np.zeros_like(self.X_col)
        dout_col = output_error.transpose(2, 3, 0, 1).ravel()
        dX = self.dpool(dX_col, dout_col, self.max_idx)
        dX = col2im(dX_col, (n*d, h, w, 1), self.size, pad=0, stride=self.stride)
        dX = dX.reshape(self.X_shape)
        return dX


class MaxPoling(Poling2D):

    def __init__(self, region_shape):
        self.region_h, self.region_w = region_shape

    def pool(X_col):
        max_idx = np.argmax(X_col, axis=0)
        out = X_col[max_idx, range(max_idx.size)]
        return out, max_idx

    def dpool(dX_col, dout_col, pool_cache):
        dX_col[pool_cache, range(dout_col.size)] = dout_col
        return dX_col

    def forward(self, input_data):
        self.X_input = input_data
        _, self.input_h, self.input_w, self.input_f = input_data.shape

        self.out_h = self.input_h // self.region_h
        self.out_w = self.input_w // self.region_w
        output = np.zeros((self.out_h, self.out_w, self.input_f))

        for image, i, j in self.iterate_regions():
            output[i, j] = np.amax(image)
        return output

    def backward(self, output_error, lr):
        n, w, h, d = self.X_shape

        dX_col = np.zeros_like(self.X_shape)
        dout_col = output_error.transpose(2, 3, 0, 1).ravel()

        dX = self.dpool(dX_col, dout_col, self.max_idx)

        dX = col2im(dX_col, (n * d, 1, h, w), pad = 0, stride = self.stride)
        dX = dX.reshape(self.X_shape)

    def iterate_regions(self):
        for i in range(self.out_h):
            for j in range(self.out_w):
                image = self.X_input[(i * self.region_h) : (i * self.region_h + 2), (j * self.region_h) : (j * self.region_h + 2)]
                yield image, i, j