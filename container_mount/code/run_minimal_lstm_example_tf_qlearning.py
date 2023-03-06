#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""created 04.01.2022
@author: holger mohr
"""

import numpy as np
import tensorflow as tf


class SimpleEnv:
    
    def __init__(self, rng):
        
        self.rng = rng
        self.screen = np.array([1, 0])
        self.prev_screen = np.array([1, 0])
        self.frame_no = 0
    
    def __call__(self, action):
        
        if all(self.screen == self.prev_screen):
            reward_vec = np.array([1.0, 0.0])
        else:
            reward_vec = np.array([0.0, 2.0])
        
        reward = reward_vec[action]
        
        self.prev_screen = self.screen
        
        draw_screen = self.rng.integers(2)
        
        if draw_screen == 0:
            screen = np.array([1, 0])
        elif draw_screen == 1:
            screen = np.array([0, 1])
        
        self.screen = screen
        self.frame_no += 1
        
        return reward


class QAgentManual:
    
    def __init__(self, rng, eps_rand, learning_rate):
        
        self.rng = rng
        self.eps_rand = eps_rand
        self.learning_rate = learning_rate
        self.Q = np.array([[0.5, 0.5], [0.5, 0.5]])
        self.reward_pred = self.Q[0,0]
        self.screen = np.array([1, 0])
        self.action = 0
    
    def __call__(self, screen):  # forward pass
        
        self.screen = screen
        screen_tensor = np.array(screen, dtype=float)
        q_vec = np.matmul(self.Q, screen_tensor)
        action = self.get_response_from_q(q_vec)
        self.action = action
        self.reward_pred = q_vec[action]
        return action
    
    def get_response_from_q(self, q_vec):
        
        if self.rng.random() < self.eps_rand:
            
            action = self.rng.integers(2)
            
        else:
            
            if q_vec[0] == q_vec[1]:
                action = self.rng.integers(2)
            else:
                action = np.argmax(q_vec)
        
        self.eps_rand = max(self.eps_rand - 0.01, 0)
        
        return action
    
    def qlearning_manual(self, reward):
        
        reward_torch = np.array(reward, dtype=float)
        
        if all(self.screen == np.array([1, 0])) and self.action == 0:
            self.Q[0,0] -= self.learning_rate*(self.reward_pred - reward_torch)
        elif all(self.screen == np.array([1, 0])) and self.action == 1:
            self.Q[1,0] -= self.learning_rate*(self.reward_pred - reward_torch)
        elif all(self.screen == np.array([0, 1])) and self.action == 0:
            self.Q[0,1] -= self.learning_rate*(self.reward_pred - reward_torch)
        elif all(self.screen == np.array([0, 1])) and self.action == 1:
            self.Q[1,1] -= self.learning_rate*(self.reward_pred - reward_torch)


class QAgentTF:

    def __init__(self, rng, eps_rand, learning_rate):
        
        self.rng = rng
        self.eps_rand = eps_rand
        self.learning_rate = learning_rate
        self.screen = np.array([1, 0])
        self.action = 0
        #initializer = tf.keras.initializers.Constant(0.5)
        #self.model = tf.keras.models.Sequential()
        #self.model.add(tf.keras.layers.LSTM(2))
        layer_input = tf.keras.Input(shape=(1,2), batch_size=1)
        layer_lstm = tf.keras.layers.LSTM(4, stateful=True, activation=None, recurrent_activation=None, trainable=False)(layer_input)
        layer_hidden = tf.keras.layers.Dense(4, activation='relu', trainable=False)(layer_lstm)
        layer_out = tf.keras.layers.Dense(2, activation='relu', trainable=False)(layer_hidden)
        #self.model = tf.keras.Model(inputs=layer_input, outputs=[layer_out, layer_hidden])
        self.model = tf.keras.Model(inputs=layer_input, outputs=layer_out)
        #self.model.add(tf.keras.layers.Dense(4, input_shape=(4,), activation='relu'))
        #self.model.add(tf.keras.layers.Dense(2, activation='relu'))
        self.model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate), loss=tf.keras.losses.MeanSquaredError())
        self.reward_pred = 0.5
        
        W_f = np.zeros((4,2), dtype=np.float32)
        U_f = np.zeros((4,4), dtype=np.float32)
        b_f = np.zeros((4,), dtype=np.float32)
        
        W_i = np.zeros((4,2), dtype=np.float32)
        U_i = np.zeros((4,4), dtype=np.float32)
        b_i = np.ones((4,), dtype=np.float32)
        
        W_c = np.zeros((4,2), dtype=np.float32)
        W_c[0,0] = 1
        W_c[1,1] = 1
        U_c = np.zeros((4,4), dtype=np.float32)
        U_c[2,0] = 1
        U_c[3,1] = 1
        b_c = np.zeros((4,), dtype=np.float32)
        
        W_o = np.zeros((4,2), dtype=np.float32)
        U_o = np.zeros((4,4), dtype=np.float32)
        b_o = np.ones((4,), dtype=np.float32)
        
        W_all = np.concatenate((W_i, W_f, W_c, W_o), axis=0).transpose()
        U_all = np.concatenate((U_i, U_f, U_c, U_o), axis=0).transpose()
        b_all = np.concatenate((b_i, b_f, b_c, b_o), axis=0)
        
        self.model.layers[1].set_weights([W_all, U_all, b_all])
        self.model.layers[2].set_weights([np.array([[1,0,1,0],[0,1,0,1],[1,0,0,1],[0,1,1,0]], dtype=np.float32), np.array([-1,-1,-1,-1], dtype=np.float32)])
        self.model.layers[3].set_weights([np.array([[1,0],[1,0],[0,2],[0,2]], dtype=np.float32), np.array([0,0], dtype=np.float32)])
        
        #self.lstm_state = self.model.layers[0].get_initial_state(batch_size=1, dtype=tf.float32)
        #print('lstm initial state:')
        #print(self.lstm_state)
        #self.lstm_state = None
        #self.lstm_standalone = tf.keras.layers.LSTM(4, stateful=True, return_state=True)
        
        model_out = self.model(self.screen[None,None,:].astype(float)) # LSTM version with time dimension (one more None)
        #model_out = self.model(self.screen[None,:].astype(float))
        #self.q_vec = model_out[0]
        self.q_vec = model_out
        
        self.h_old = np.array([0., 0., 0., 0.])
        self.c_old = np.array([0., 0., 0., 0.])

    
    
    def __call__(self, screen):
        
        def lstm_holger(x, W, U, b, h, c):
            
            print('x shape:')
            print(x.shape)
            print('b shape:')
            print(b.shape)
            h = np.squeeze(h.transpose())
            print('h shape:')
            print(h.shape)
            W_i = W[:,0:4].transpose()
            print('W_i shape:')
            print(W_i.shape)
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
            
            print('matmul shape:')
            print((np.matmul(W_i,x) + np.matmul(U_i,h) + b_i).shape)
            
            #i = np.squeeze(tf.math.sigmoid(tf.constant(np.matmul(W_i,x) + np.matmul(U_i,h) + b_i)).numpy())
            #f = np.squeeze(tf.math.sigmoid(tf.constant(np.matmul(W_f,x) + np.matmul(U_f,h) + b_f)).numpy())
            #c_tilde = np.squeeze(tf.math.tanh(tf.constant(np.matmul(W_c,x) + np.matmul(U_c,h) + b_c)).numpy())
            #o = np.squeeze(tf.math.sigmoid(tf.constant(np.matmul(W_o,x) + np.matmul(U_o,h) + b_o)).numpy())
            
            #c = np.squeeze(np.multiply(f, c) + np.multiply(i, c_tilde))
            #h = np.multiply(o, np.squeeze(tf.math.tanh(tf.constant(c)).numpy()))
            
            i = np.squeeze(tf.identity(tf.constant(np.matmul(W_i,x) + np.matmul(U_i,h) + b_i)).numpy())
            f = np.squeeze(tf.identity(tf.constant(np.matmul(W_f,x) + np.matmul(U_f,h) + b_f)).numpy())
            c_tilde = np.squeeze(tf.identity(tf.constant(np.matmul(W_c,x) + np.matmul(U_c,h) + b_c)).numpy())
            o = np.squeeze(tf.identity(tf.constant(np.matmul(W_o,x) + np.matmul(U_o,h) + b_o)).numpy())
            
            print('i:')
            print(i)
            print('f:')
            print(f)
            print('c_tilde:')
            print(c_tilde)
            print('o:')
            print(o)
            
            c = np.squeeze(np.multiply(f, c) + np.multiply(i, c_tilde))
            h = np.multiply(o, np.squeeze(tf.identity(tf.constant(c)).numpy()))
            
            return h, c
        
        self.screen = screen
        model_out = self.model(self.screen[None,None,:].astype(float)) # LSTM version
        #model_out = self.model(self.screen[None,:].astype(float))
        #self.q_vec = model_out[0]
        self.q_vec = model_out
        #print('q_vec:')
        #print(self.q_vec)
        action = self.get_response_from_q()
        self.action = action
        self.reward_pred = self.q_vec[0, action]
        print('weights layer 1:')
        print(self.model.layers[1].get_weights())
        print('weights layer 1 1:')
        print(self.model.layers[1].get_weights()[0].shape)
        print('weights layer 1 2:')
        print(self.model.layers[1].get_weights()[1].shape)
        print('weights layer 1 3:')
        print(self.model.layers[1].get_weights()[2].shape)
        print('weights layer 2:')
        print(self.model.layers[2].get_weights())
        print('weights layer 3:')
        print(self.model.layers[3].get_weights())
        print('output layer 3:')
        print(model_out[0])
        #print('output layer 1:')
        #print(model_out[1])
        #lstm_output, lstm_state_1, lstm_state_2 = self.lstm_standalone(self.screen[None,None,:].astype(float))
        #print('lstm_output:')
        #print(lstm_output)
        #print('lstm_state_1:')
        #print(lstm_state_1)
        #print('lstm_state_2:')
        #print(lstm_state_2)
        
        h_new, c_new = lstm_holger(screen, 
                           self.model.layers[1].get_weights()[0], 
                           self.model.layers[1].get_weights()[1], 
                           self.model.layers[1].get_weights()[2], 
                           self.h_old, 
                           self.c_old)
        
        print('holger h:');
        print(h_new)
        print('holger c:');
        print(c_new)
        
        self.h_old = h_new
        self.c_old = c_new
        
        return action
    
    def get_response_from_q(self):
        
        if self.rng.random() < self.eps_rand:
            action = self.rng.integers(2)
        else:
            if self.q_vec[0,0] == self.q_vec[0,1]:
                action = self.rng.integers(2)
            else:
                action = np.argmax(self.q_vec)
            
        self.eps_rand = max(self.eps_rand - 0.01, 0)
        return action
    
    def qlearning_tf(self, reward):
        
        #self.q_vec = self.model(self.screen[None,None,:].astype(float))
        reward_array = self.q_vec.numpy()
        reward_array[0, self.action] = reward
        self.model.fit(self.screen[None,None,:].astype(float), reward_array, batch_size=1, epochs=1, steps_per_epoch=1) # LSTM version
        #self.model.fit(self.screen[None,:].astype(float), reward_array, batch_size=1, epochs=1, steps_per_epoch=1)


"""
class QAgentTF_fit:

    def __init__(self, rng, eps_rand, learning_rate):
        
        self.rng = rng
        self.eps_rand = eps_rand
        self.learning_rate = learning_rate
        self.screen = np.array([1, 0])
        self.action = 0
        initializer = tf.keras.initializers.Constant(0.5)
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Dense(2, input_shape=(2,), 
                                             activation=None, 
                                             use_bias=False,
                                             kernel_initializer=initializer))
        self.model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate==self.learning_rate), 
                           loss=tf.keras.losses.MeanSquaredError())
        self.reward_pred = self.model.layers[0].get_weights()[0][0,0]
        self.q_vec = np.array([0.0, 0.0])
        
    def __call__(self, screen):
        
        self.screen = screen
        self.q_vec = np.squeeze(self.model.predict(screen[None,:]))
        action = self.get_response_from_q()
        self.action = action
        self.reward_pred = self.q_vec[action]
        return action
    
    def get_response_from_q(self):
        
        if self.rng.random() < self.eps_rand:
            action = self.rng.integers(2)
        else:
            if self.q_vec[0] == self.q_vec[1]:
                action = self.rng.integers(2)
            else:
                action = np.argmax(self.q_vec)
            
        self.eps_rand = max(self.eps_rand - 0.01, 0)
        return action
    
    def qlearning_tf(self, reward):
        
        q_vec_temp = self.model(self.screen[None,:])
        reward_array = q_vec_temp.numpy()
        reward_array[0, self.action] = reward
        self.model.fit(self.screen[None,:], reward_array, batch_size=1, epochs=1, steps_per_epoch=1)
"""

def main():
    
    rng_manual = np.random.default_rng(1)
    rng_tf = np.random.default_rng(1)
    #rng_tf_fit = np.random.default_rng(1)
    
    env_manual = SimpleEnv(rng_manual)
    env_tf = SimpleEnv(rng_tf)
    #env_tf_fit = SimpleEnv(rng_tf_fit)
    
    frame_no_manual = env_manual.frame_no
    frame_no_tf = env_tf.frame_no
    #frame_no_tf_fit = env_tf_fit.frame_no
    
    n_frames = 1000
    eps_rand = 1.0
    learning_rate = 0.1
    
    action_manual = 0
    action_tf = 0
    #action_tf_fit = 0
    
    total_reward_manual = 0.0
    total_reward_tf = 0.0
    
    q_agent_manual = QAgentManual(rng_manual, eps_rand, learning_rate)
    q_agent_tf = QAgentTF(rng_tf, eps_rand, learning_rate)
    #q_agent_tf_fit = QAgentTF_fit(rng_tf_fit, eps_rand, learning_rate)
    
    while frame_no_manual < n_frames:
        
        print(' ')
        print('************************************')
        
        frame_no_manual = env_manual.frame_no
        screen_old_manual = env_manual.screen
        reward_manual = env_manual(action_manual)
        total_reward_manual = total_reward_manual + reward_manual
        
        print(' ')
        print('QAgentManual:')
        print('frame no: ' + str(frame_no_manual))
        print('eps_rand: ' + str(q_agent_manual.eps_rand))
        print('screen: ' + str(screen_old_manual))
        print('action: ' + str(action_manual))
        print('reward: ' + str(reward_manual))
        print('total reward: ' + str(total_reward_manual))
        
        frame_no_tf = env_tf.frame_no
        screen_old_tf = env_tf.screen
        reward_tf = env_tf(action_tf)
        total_reward_tf = total_reward_tf + reward_tf
        
        print(' ')
        print('QAgentTF:')
        print('frame no: ' + str(frame_no_tf))
        print('eps_rand: ' + str(q_agent_tf.eps_rand))
        print('screen: ' + str(screen_old_tf))
        print('action: ' + str(action_tf))
        print('reward: ' + str(reward_tf))
        print('total reward: ' + str(total_reward_tf))
        """
        frame_no_tf_fit = env_tf_fit.frame_no
        screen_old_tf_fit = env_tf_fit.screen
        reward_tf_fit = env_tf_fit(action_tf_fit)
        
        print(' ')
        print('QAgentTF_fit:')
        print('frame no: ' + str(frame_no_tf_fit))
        print('eps_rand: ' + str(q_agent_tf_fit.eps_rand))
        print('screen: ' + str(screen_old_tf_fit))
        print('action: ' + str(action_tf_fit))
        print('reward: ' + str(reward_tf_fit))
        """
        
        """ learning """
        
        print(' ')
        print('Q:')
        print(q_agent_manual.Q)
        print(' ')
        print('QAgentTF:')
        #print(q_agent_tf.model.layers[0].get_weights()[0])
        #print(' ')
        #print('QAgentTF_fit:')
        #print(q_agent_tf_fit.model.layers[0].get_weights()[0])
        
        q_agent_manual.qlearning_manual(reward_manual)
        q_agent_tf.qlearning_tf(reward_tf)
        #q_agent_tf_fit.qlearning_tf(reward_tf_fit)
        
        
        """ forward pass """
        
        screen_new_manual = env_manual.screen
        #screen_prev_manual = env_manual.prev_screen
        #screen_both_manual = np.concatenate((screen_new_manual, screen_prev_manual))
        #print(screen_both_manual)
        
        screen_new_tf = env_tf.screen
        #screen_prev_tf = env_tf.prev_screen
        #screen_both_tf = np.concatenate((screen_new_tf, screen_prev_tf))
        #screen_new_tf_fit = env_tf_fit.screen
        
        action_manual = q_agent_manual(screen_new_manual)
        action_tf = q_agent_tf(screen_new_tf)
        #action_tf_fit = q_agent_tf_fit(screen_new_tf_fit)
        
        frame_no_manual = env_manual.frame_no


if __name__ == '__main__':
    main()
