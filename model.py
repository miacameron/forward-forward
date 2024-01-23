import numpy as np
from params import get_params
from utils import *

class ForwardForward:

    def __init__(self, par : dict):

        self.par = par
        self.learn_positive = True
        self.learn_negative = False
        self.epoch = -1
        self.__init_weights()

        self.P1_log = np.zeros((1))
        self.P2_log = np.zeros((1))
        self.P3_log = np.zeros((1))


    def __init_weights(self):

        p1 = self.par['L1_par']
        self.L1 = np.random.uniform(-0.05, 0.05, (p1['grid_dim']**2, p1['out_channels'], p1['rf_dim']**2 * p1['rf_channels']))
        self.dW1 = np.zeros((p1['grid_dim']**2, p1['out_channels'], p1['rf_dim']**2 * p1['rf_channels']))
        
        p2 = self.par['L2_par']
        self.L2 = np.random.normal(-0.05, 0.05, (p2['grid_dim']**2, p2['out_channels'], p2['rf_dim']**2 * p2['rf_channels']))
        self.dW2 = np.zeros((p2['grid_dim']**2, p2['out_channels'], p2['rf_dim']**2 * p2['rf_channels']))
        
        p3 = self.par['L3_par']
        self.L3 = np.random.normal(-0.05, 0.05, (p3['grid_dim']**2, p3['out_channels'], p3['rf_dim']**2 * p3['rf_channels']))
        self.dW3 = np.zeros((p3['grid_dim']**2, p3['out_channels'], p3['rf_dim']**2 * p3['rf_channels']))
        
    
    def __convolve(self, x, layer):

        if (layer == 0):
            assert x.shape == (28,28)
            p = self.par['L1_par']
            W = self.L1
            log = self.P1_log
        elif (layer == 1):
            assert x.shape == (4,4,128)
            p = self.par['L2_par']
            W = self.L2
            log = self.P2_log
        elif (layer == 2):
            assert x.shape == (3,3,220)
            p = self.par['L3_par']
            W = self.L3
            log = self.P3_log

        dW = np.zeros((p['grid_dim']**2, p['out_channels'], p['rf_dim']**2 * p['rf_channels']))
        y = np.zeros((p['grid_dim']**2, p['out_channels']))

        s = p['rf_stride']
        r = p['rf_dim']

        for i in range(p['grid_dim']**2): # for every location in the grid

            i_x = int(i / p['grid_dim']) * s # in (0,1)
            i_y = int(i % p['grid_dim']) * s
                
            x_ = x[i_x : i_x + r, i_y : i_y + r] # limit to current receptive field
            x_ = x_.reshape(p['rf_dim']**2 * p['rf_channels']) # flatten
            y[i,:] = ReLU(np.dot(W[i,:], x_)) # dot product with L1 weight matrix
            P = self.P(y[i,:])                

            if (self.learn_positive):
                log[self.epoch] += P # record goodness of layer 1
                dW[i,:] -= 2 * P * np.outer(y[i], x_)

            elif (self.learn_negative):
                dW[i,:] += 2 * P * np.outer(y[i], x_)
          
        y = y.reshape(p['grid_dim'], p['grid_dim'], p['out_channels'])
        if (np.sum(y) != 0): # layer-normalization
            y = y / np.linalg.norm(y)
        return y, dW

    def P(self, y):
        return logistic(np.sum(np.power(y,2)) - self.par['thresh'])

    def forward_pass(self, x0):
        x1, dW1 = self.__convolve(x0,layer=0)
        x2, dW2 = self.__convolve(x1,layer=1)
        x3, dW3 = self.__convolve(x2,layer=2)
        return x1, x2, x3, dW1, dW2, dW3

    def update_weights(self, dW1, dW2, dW3):
        self.L1 += self.par['lr'] * dW1
        self.L2 += self.par['lr'] * dW2
        self.L3 += self.par['lr'] * dW3

    def learn_batch(self, X_pos, X_neg):

        batch_size = 100
        pos_examples = X_pos.shape[0]
        neg_examples = X_neg.shape[0]
        assert (pos_examples == neg_examples)
        
        for i in range(int(pos_examples / batch_size) - 1):
            #print("Batch : {}".format(i))
            dW1_sum = 0
            dW2_sum = 0
            dW3_sum = 0

            for j in range(batch_size):
                self.learn_positive = True
                self.learn_negative = False
                x0 = X_pos[i*batch_size + j]
                x1, x2, x3, dW1, dW2, dW3 = self.forward_pass(x0)
                dW1_sum += dW1; dW2_sum += dW2; dW3_sum += dW3 

            for j in range(batch_size):
                self.learn_positive = False
                self.learn_negative = True
                x0 = X_neg[i*batch_size + j]
                x1, x2, x3, dW1, dW2, dW3 = self.forward_pass(x0)
                dW1_sum += dW1; dW2_sum += dW2; dW3_sum += dW3 
            
            self.update_weights(dW1_sum, dW2_sum, dW3_sum)

        return     
            
    def learn(self, X_pos, X_neg, epochs):

        self.P1_log = np.zeros((epochs))
        self.P2_log = np.zeros((epochs))
        self.P3_log = np.zeros((epochs))

        for e in range(epochs):
            print("Epoch : {}".format(e))
            self.epoch = e
            self.learn_batch(X_pos, X_neg)