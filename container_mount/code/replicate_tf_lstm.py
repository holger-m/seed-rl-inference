#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""created 04.01.2022
@author: holger mohr
"""

import numpy as np
import tensorflow as tf

def numpy_sigmoid(z):
    
    sig = 1 / (1 + np.exp(-z))
    
    return sig


def lstm_holger(x, W, U, b, h, c):
    
    #print('x shape:')
    #print(x.shape)
    #print('b shape:')
    #print(b.shape)
    h = np.squeeze(h.transpose())
    #print('h shape:')
    #print(h.shape)
    W_i = W[:,0:4].transpose()
    #print('W_i shape:')
    #print(W_i.shape)
    W_f = W[:,4:8].transpose()
    W_c = W[:,8:12].transpose()
    W_o = W[:,12:16].transpose()
    
    U_i = U[:,0:4].transpose()
    U_f = U[:,4:8].transpose()
    U_c = U[:,8:12].transpose()
    U_o = U[:,12:16].transpose()
    
    b_i = b[0:4]
    b_f = b[4:8]
    b_c = b[8:12]
    b_o = b[12:16]
    
    #print('matmul shape:')
    #print((np.matmul(W_i,x) + np.matmul(U_i,h) + b_i).shape)
    
    #i = np.squeeze(tf.math.sigmoid(tf.constant(np.matmul(W_i,x) + np.matmul(U_i,h) + b_i)).numpy())
    #f = np.squeeze(tf.math.sigmoid(tf.constant(np.matmul(W_f,x) + np.matmul(U_f,h) + b_f)).numpy())
    #c_tilde = np.squeeze(tf.math.tanh(tf.constant(np.matmul(W_c,x) + np.matmul(U_c,h) + b_c)).numpy())
    #o = np.squeeze(tf.math.sigmoid(tf.constant(np.matmul(W_o,x) + np.matmul(U_o,h) + b_o)).numpy())
    
    #c = np.squeeze(np.multiply(f, c) + np.multiply(i, c_tilde))
    #h = np.multiply(o, np.squeeze(tf.math.tanh(tf.constant(c)).numpy()))
    
    #print('W_i*x:')
    #print(np.matmul(W_i,x))
    #print('W_c*x:')
    #print(np.matmul(W_c,x))
    
    i = np.squeeze(tf.sigmoid(tf.constant(np.matmul(W_i,x) + np.matmul(U_i,h) + b_i)).numpy())
    f = np.squeeze(tf.sigmoid(tf.constant(np.matmul(W_f,x) + np.matmul(U_f,h) + b_f)).numpy())
    c_tilde = np.squeeze(tf.tanh(tf.constant(np.matmul(W_c,x) + np.matmul(U_c,h) + b_c)).numpy())
    o = np.squeeze(tf.sigmoid(tf.constant(np.matmul(W_o,x) + np.matmul(U_o,h) + b_o)).numpy())
    
    #print('i:')
    #print(i)
    #print('f:')
    #print(f)
    #print('c_tilde:')
    #print(c_tilde)
    #print('o:')
    #print(o)
    c = np.squeeze(np.multiply(f, c) + np.multiply(i, c_tilde))
    h = np.multiply(o, np.squeeze(tf.tanh(tf.constant(c)).numpy()))
    #print('c:')
    #print(c)
    
    return h, c


def main():
    
    rng = np.random.default_rng()
    layer_input = tf.keras.Input(shape=(1,2), batch_size=1)
    layer_lstm = tf.keras.layers.LSTM(4, activation='tanh', recurrent_activation='sigmoid',
                                      return_state=True, stateful=True)(layer_input)
    model = tf.keras.Model(inputs=layer_input, outputs=layer_lstm)
    
    #model.layers[1].set_weights([rng.random((2,16), dtype=np.float32), rng.random((4,16), dtype=np.float32), np.array([0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=np.float32)])
    
    #print(model.get_weights())
    #print('weights layer 1 1:')
    #print(model.get_weights()[0].shape)
    #print('weights layer 1 2:')
    #print(model.get_weights()[1].shape)
    #print('weights layer 1 3:')
    #print(model.get_weights()[2].shape)
    
    h_old = np.array([0., 0., 0., 0.], dtype=np.float32)
    c_old = np.array([0., 0., 0., 0.], dtype=np.float32)
    s_old = [h_old, c_old]
    
    n_timepoints = 2
    
    for time_t in range(n_timepoints):
        
        print(' ')
        print(' ')
        print(' ')
        print(' ')
        print('round ' + str(time_t) + ':')
        
        x = rng.random((2,), dtype=np.float32)
        
        tf_out = model(x[None,None,:])
        
        h_new, c_new = lstm_holger(x, 
                                   model.layers[1].get_weights()[0], 
                                   model.layers[1].get_weights()[1], 
                                   model.layers[1].get_weights()[2], 
                                   h_old, 
                                   c_old)
        
        print('tf out 0:')
        print(tf_out[0][0].numpy())
        print('h_new:')
        print(h_new)
        #print('tf out 1:')
        #print(tf_out[1])
        print(' ')
        print('tf out 2:')
        print(tf_out[2][0].numpy())
        print('c_new:')
        print(c_new)
        
        h_diff = tf_out[0] - h_new
        c_diff = tf_out[2] - c_new
        
        print(' ')
        print('h_diff:')
        print(h_diff[0].numpy())
        print('c_diff:')
        print(c_diff[0].numpy())
        
        h_old = h_new
        c_old = c_new


if __name__ == '__main__':
    main()
