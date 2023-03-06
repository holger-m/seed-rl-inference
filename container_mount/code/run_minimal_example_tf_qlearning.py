#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""created 04.04.2022
@author: holger mohr
"""

import numpy as np
import tensorflow as tf


class SimpleEnv:
    
    def __init__(self, rng):
        
        self.rng = rng
        self.screen = np.array([1, 0])
        self.frame_no = 0
    
    def __call__(self, action):
        
        if all(self.screen == np.array([1, 0])):
            reward_vec = np.array([1.0, 0.0])
        elif all(self.screen == np.array([0, 1])):
            reward_vec = np.array([0.0, 2.0])
        
        reward = reward_vec[action]
        
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
        initializer = tf.keras.initializers.Constant(0.5)
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Dense(2, input_shape=(2,), 
                                             activation=None, 
                                             use_bias=False,
                                             kernel_initializer=initializer))
        self.model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate))
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
        
        with tf.GradientTape() as tape:
            q_vec_temp = self.model(self.screen[None,:])
            loss = 0.5*(reward - q_vec_temp[0, self.action])**2
        grad = tape.gradient(loss, self.model.trainable_variables)
        self.model.trainable_variables[0].assign_sub(self.learning_rate*grad[0])


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
        self.model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate), 
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
        #self.model.fit(self.screen[None,:], reward_array, batch_size=1, epochs=1, steps_per_epoch=1)
        self.model.fit(self.screen[None,:], reward_array)


def main():
    
    rng_manual = np.random.default_rng(1)
    rng_tf = np.random.default_rng(1)
    rng_tf_fit = np.random.default_rng(1)
    
    env_manual = SimpleEnv(rng_manual)
    env_tf = SimpleEnv(rng_tf)
    env_tf_fit = SimpleEnv(rng_tf_fit)
    
    frame_no_manual = env_manual.frame_no
    frame_no_tf = env_tf.frame_no
    frame_no_tf_fit = env_tf_fit.frame_no
    
    n_frames = 100
    eps_rand = 1.0
    learning_rate = 0.1
    
    action_manual = 0
    action_tf = 0
    action_tf_fit = 0
    
    q_agent_manual = QAgentManual(rng_manual, eps_rand, learning_rate)
    q_agent_tf = QAgentTF(rng_tf, eps_rand, learning_rate)
    q_agent_tf_fit = QAgentTF_fit(rng_tf_fit, eps_rand, learning_rate)
    
    while frame_no_manual < n_frames:
        
        print(' ')
        print('************************************')
        
        frame_no_manual = env_manual.frame_no
        screen_old_manual = env_manual.screen
        reward_manual = env_manual(action_manual)
        
        print(' ')
        print('QAgentManual:')
        print('frame no: ' + str(frame_no_manual))
        print('eps_rand: ' + str(q_agent_manual.eps_rand))
        print('screen: ' + str(screen_old_manual))
        print('action: ' + str(action_manual))
        print('reward: ' + str(reward_manual))
        
        frame_no_tf = env_tf.frame_no
        screen_old_tf = env_tf.screen
        reward_tf = env_tf(action_tf)
        
        print(' ')
        print('QAgentTF:')
        print('frame no: ' + str(frame_no_tf))
        print('eps_rand: ' + str(q_agent_tf.eps_rand))
        print('screen: ' + str(screen_old_tf))
        print('action: ' + str(action_tf))
        print('reward: ' + str(reward_tf))
        
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
        
        
        """ learning """
        
        print(' ')
        print('Q:')
        print(q_agent_manual.Q)
        print(' ')
        print('QAgentTF:')
        print(q_agent_tf.model.layers[0].get_weights()[0])
        print(' ')
        print('QAgentTF_fit:')
        print(q_agent_tf_fit.model.layers[0].get_weights()[0])
        
        q_agent_manual.qlearning_manual(reward_manual)
        q_agent_tf.qlearning_tf(reward_tf)
        q_agent_tf_fit.qlearning_tf(reward_tf_fit)
        
        
        """ forward pass """
        
        screen_new_manual = env_manual.screen
        screen_new_tf = env_tf.screen
        screen_new_tf_fit = env_tf_fit.screen
        
        action_manual = q_agent_manual(screen_new_manual)
        action_tf = q_agent_tf(screen_new_tf)
        action_tf_fit = q_agent_tf_fit(screen_new_tf_fit)
        
        frame_no_manual = env_manual.frame_no


if __name__ == '__main__':
    main()
