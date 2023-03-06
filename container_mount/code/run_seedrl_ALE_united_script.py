#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""created 15.02.2022
@author: holger mohr
Example usage: ./run_seedrl_ALE_united_script.py -r /workspace/container_mount/data/ROMs/enduro.bin -cp 0/ckpt-98 -sn 1 -sp /workspace/container_mount/ramdisk
"""

import argparse
import os
import numpy as np
from ale_python_interface import ALEInterface
import pygame

from seed_rl.atari import networks
from seed_rl.common import utils
import math
import tensorflow as tf
import cv2
import gym


class RunALE:
    
    def __init__(self, rom_path, noop, eps_greedy, rng):
        
        self.noop = noop
        self.eps_greedy = eps_greedy
        self.rng = rng
        self.display_width = 800
        self.display_height = 600
        self.noop_frame_count_15hz = 0
        
        # init ALE
        self.ale = ALEInterface()
        max_frames_per_episode = self.ale.getInt(b'max_num_frames_per_episode')
        print('ALE max frames per episode: ' + str(max_frames_per_episode))
        
        random_int = self.rng.integers(2^32)
        self.ale.setInt(b'random_seed', random_int)
        ale_seed = self.ale.getInt(b'random_seed')
        print('ALE random seed: ' + str(ale_seed))
        
        self.ale.setFloat(b'repeat_action_probability', 0.0)
        action_repeat_prob = self.ale.getFloat(b'repeat_action_probability')
        print('ALE action repeat prob.: ' + str(action_repeat_prob))
        
        self.ale.loadROM(str.encode(rom_path))
        legal_actions = self.ale.getMinimalActionSet()
        print('ALE legal actions:')
        print(legal_actions)
        
        (screen_width, screen_height) = self.ale.getScreenDims()
        self.screen_width = screen_width
        self.screen_height = screen_height
        print('ALE screen dims: ' + str(screen_width) + '/' + str(screen_height))
        print(' ')
        
        # init Pygame
        pygame.display.init()
        pygame.font.init()
        self.pygame_display = pygame.display.set_mode((self.display_width, self.display_height))
        pygame.display.set_caption('ALE Display')
        self.game_surface = pygame.Surface((screen_width, screen_height))
        pygame.mouse.set_visible(False)
        self.clock = pygame.time.Clock()
        
        # init variables
        self.screen_vec_RGB_1 = np.empty((screen_height, screen_width, 3), dtype=np.uint8)
        self.screen_vec_RGB_2 = np.empty((screen_height, screen_width, 3), dtype=np.uint8)
        self.screen_vec_Gray_1 = np.empty((screen_height, screen_width), dtype=np.uint8)
        self.screen_vec_Gray_2 = np.empty((screen_height, screen_width), dtype=np.uint8)
    
    def ale_15hz(self, action):
        
        reward_sum = 0
        episode_end_flag = False
        
        reward = self.ale.act(action)
        reward_sum += reward
        if(self.ale.game_over()):
            episode_end_flag = True
        
        reward = self.ale.act(action)
        reward_sum += reward
        if(self.ale.game_over()):
            episode_end_flag = True
        
        reward = self.ale.act(action)
        reward_sum += reward
        if(self.ale.game_over()):
            episode_end_flag = True
        self.ale.getScreenRGB(self.screen_vec_RGB_1)
        self.ale.getScreenGrayscale(self.screen_vec_Gray_1)
        
        reward = self.ale.act(action)
        reward_sum += reward
        if(self.ale.game_over()):
            episode_end_flag = True
        self.ale.getScreenRGB(self.screen_vec_RGB_2)
        self.ale.getScreenGrayscale(self.screen_vec_Gray_2)
        
        screen_R_max = np.amax(np.dstack((self.screen_vec_RGB_1[:,:,0], 
                                          self.screen_vec_RGB_2[:,:,0])), axis=2)
        screen_G_max = np.amax(np.dstack((self.screen_vec_RGB_1[:,:,1], 
                                          self.screen_vec_RGB_2[:,:,1])), axis=2)
        screen_B_max = np.amax(np.dstack((self.screen_vec_RGB_1[:,:,2], 
                                          self.screen_vec_RGB_2[:,:,2])), axis=2)
        self.screen_RGB_max = np.dstack((screen_R_max, screen_G_max, screen_B_max))
        
        screen_Gray_max = np.amax(np.dstack((self.screen_vec_Gray_1, 
                                             self.screen_vec_Gray_2)), axis=2)
        
        if episode_end_flag:
            self.noop_frame_count_15hz = 0
            self.ale.reset_game()
        
        return screen_Gray_max, reward_sum, episode_end_flag
    
    def pygame_step(self):
        
        pygame.event.pump()
        self.pygame_display.fill((0,0,0))
        numpy_surface = np.frombuffer(self.game_surface.get_buffer(),dtype=np.uint8)
        
        screen_RGB_reversed = np.reshape(np.dstack((self.screen_RGB_max[:,:,2], 
                                                    self.screen_RGB_max[:,:,1], 
                                                    self.screen_RGB_max[:,:,0], 
                                                    np.zeros((self.screen_height,
                                                              self.screen_width), 
                                                    dtype=np.uint8))), 210*160*4)
        numpy_surface[:] = screen_RGB_reversed
        del numpy_surface
        self.pygame_display.blit(pygame.transform.scale(self.game_surface, 
                                                       (self.display_width, 
                                                        self.display_height)),
                                                       (0,0))
        pygame.display.flip()
        #self.clock.tick(15)
    
    def __call__(self, action):
        
        self.noop_frame_count_15hz += 1
        
        u_greedy = self.rng.random()
        if u_greedy <= self.eps_greedy:
            action = self.rng.integers(18)
        
        if self.noop_frame_count_15hz*4 <= self.noop: # assuming noop refers to 60hz
            action = 0
        
        action_played = action
        
        screen_Gray_max, reward_sum, episode_end_flag = self.ale_15hz(action)
        
        self.pygame_step()
        
        return screen_Gray_max, reward_sum, action_played, episode_end_flag
        
class RunSeed:
    
    def create_agent(self): # from r2d2_main.py
        return networks.DuelingLSTMDQNNet(18, (84, 84, 1), 4)
    
    def create_optimizer_fn(self, unused_final_iteration): # from r2d2_main.py
        learning_rate_fn = lambda iteration: 0.00048
        optimizer = tf.keras.optimizers.Adam(0.00048, epsilon=1e-3)
        return optimizer, learning_rate_fn
    
    def _pool_and_resize(self, screen_buffer): # from atari_preprocessing.py
        transformed_image = cv2.resize(screen_buffer, (84, 84), interpolation=cv2.INTER_LINEAR)
        int_image = np.asarray(transformed_image, dtype=np.uint8)
        return int_image
    
    def __init__(self, game_name, checkpoint_path):
        
        # code from here: https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
    
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
        
        self.agent = self.create_agent() # from learner.py
        target_agent = self.create_agent() # from learner.py
        
        optimizer, learning_rate_fn = self.create_optimizer_fn(None) # from learner.py
        
        ckpt = tf.train.Checkpoint(agent=self.agent, target_agent=target_agent, optimizer=optimizer)
        
        if game_name == 'enduro.bin':
            ckpt.restore(os.path.join('/workspace/container_mount/data/Checkpoints_from_google/Enduro', checkpoint_path))
            env = gym.make('EnduroNoFrameskip-v4', full_action_space=True) # from env.py
        elif game_name == 'breakout.bin':
            ckpt.restore(os.path.join('/workspace/container_mount/data/Checkpoints_from_google/Breakout', checkpoint_path))
            env = gym.make('BreakoutNoFrameskip-v4', full_action_space=True) # from env.py
        elif game_name == 'space_invaders.bin':
            ckpt.restore(os.path.join('/workspace/container_mount/data/Checkpoints_from_google/SpaceInvaders', checkpoint_path))
            env = gym.make('SpaceInvadersNoFrameskip-v4', full_action_space=True) # from env.py
        else:
            raise FileExistsError(game_name + ' not found!')
        
        env.seed(0)  # from learner.py: env = create_env_fn(0, FLAGS) and env.py: env.seed(task)

        env.reset()

        env.step(0)  # no action

        obs_dims = env.observation_space # from atari_preprocessing.py

        screen_buffer = np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8) # from atari_preprocessing.py

        env.observation_space.shape = (84, 84, 1) # set to these values in atari_preprocessing.py in function observation_space: shape=(screen_size,screen_size,1) with screen_size=84 in __init__ 

        # from learner.py:
        env_output_specs = utils.EnvOutput(
            tf.TensorSpec([], tf.float32, 'reward'),
            tf.TensorSpec([], tf.bool, 'done'),
            tf.TensorSpec(env.observation_space.shape, env.observation_space.dtype,
                          'observation'),
            tf.TensorSpec([], tf.bool, 'abandoned'),
            tf.TensorSpec([], tf.int32, 'episode_step'))
        
        action_specs = tf.TensorSpec([], tf.int32, 'action') # from learner.py
        agent_input_specs = (action_specs, env_output_specs) # from learner.py

        self.initial_agent_state = self.agent.initial_state(1) # from learner.py
        agent_state_specs = tf.nest.map_structure(lambda t: tf.TensorSpec(t.shape[1:], t.dtype), self.initial_agent_state) # from learner.py
        self.input_ = tf.nest.map_structure(lambda s: tf.zeros([1] + list(s.shape), s.dtype), agent_input_specs) # from learner.py

        self.current_agent_state = self.initial_agent_state
        
        self.last_15hz_screen_1_84_84_1 = np.empty([1, 84, 84, 1], dtype=np.uint8)
        
        self.episode_step_60hz = 0
    
    def __call__(self, screen_Gray_max, observed_reward, action_played, episode_end_flag):
        
        last_15hz_screen_max_84x84 = self._pool_and_resize(screen_Gray_max)
        self.last_15hz_screen_1_84_84_1[0, :, :, 0] = last_15hz_screen_max_84x84
        
        observed_reward_np = np.zeros((1,), dtype=np.float32)
        observed_reward_np[0] = observed_reward
        episode_end_flag_np = np.zeros((1,), dtype=np.bool)
        episode_end_flag_np[0] = episode_end_flag
        self.episode_step_60hz += 4
        episode_step_60hz_np = np.zeros((1,), dtype=np.int32)
        episode_step_60hz_np[0] = self.episode_step_60hz
        
        input_action, input_env = self.input_
        input_env = input_env._replace(observation=tf.convert_to_tensor(self.last_15hz_screen_1_84_84_1, dtype=np.uint8))
        input_env = input_env._replace(reward=tf.convert_to_tensor(observed_reward_np, dtype=np.float32))
        input_env = input_env._replace(done=tf.convert_to_tensor(episode_end_flag_np, dtype=np.bool))
        input_env = input_env._replace(episode_step=tf.convert_to_tensor(episode_step_60hz_np, dtype=np.int32))
        input_action = tf.convert_to_tensor(action_played, dtype=np.int32)
        self.input_ = (input_action, input_env)
        
        if episode_end_flag:
            self.current_agent_state = self.initial_agent_state
            self.episode_step_60hz = 0
            print(' ')
            print('Episode ended, set LSTM core to initial state')
        
        #current_agent_state = initial_agent_state  # test whether always initial state decreases performance
        
        agent_out = self.agent(self.input_, self.current_agent_state)
        
        AgentOutput, AgentState = agent_out
        
        self.current_agent_state = AgentState
        
        action = AgentOutput.action
        
        return action

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--ROM_path', help='ROM file path', type=str, required=True)
    parser.add_argument('-cp', '--checkpoint_path', help='Path for checkpoint files', type=str,
    required=True)
    parser.add_argument('-sn', '--session_no', help='Session number for output files', type=str,
    required=True)
    parser.add_argument('-sp', '--save_path', help='Path for output files', type=str,
    required=True)


    args = parser.parse_args()
    
    if os.path.isfile(os.path.join(args.save_path, 'reward_vec_' + args.session_no + '.csv')):
        raise FileExistsError(os.path.join(args.save_path, 'reward_vec_' 
                              + args.session_no + '.csv') + ' already exists!')
    print('Session number: ' + args.session_no)
    
    game_name = os.path.basename(args.ROM_path)
    
    rng = np.random.default_rng()
    n_frames_60hz = 108000 # must be divisible by 4
    n_frames_15hz = n_frames_60hz//4
    noop = rng.integers(30)
    eps_greedy = 0.001
    action = 0
    
    episode_no = 1
    episode_reward = 0.0
    session_reward = 0.0
    
    episode_no_vec_15hz = np.zeros(n_frames_15hz, dtype=np.int32)
    episode_reward_vec_15hz = np.zeros(n_frames_15hz, dtype=np.double)
    
    run_ale = RunALE(args.ROM_path, noop, eps_greedy, rng)
    run_seed = RunSeed(game_name, args.checkpoint_path)
    
    for loop_count_15hz in range(n_frames_15hz):
    
        screen_Gray_max, reward, action_played, episode_end_flag = run_ale(action)
        
        episode_reward += reward
        session_reward += reward
        episode_reward_vec_15hz[loop_count_15hz] = episode_reward
        episode_no_vec_15hz[loop_count_15hz] = episode_no
        
        action = run_seed(screen_Gray_max, reward, action_played, episode_end_flag)
        
        if (loop_count_15hz + 1) % 250 == 0:
            print('Total Frame Number: ' + str((loop_count_15hz + 1)*4))
            print('Episode Number: ' + str(episode_no))
            print('Episode Score: ' + str(episode_reward))
            print(' ')
        
        if episode_end_flag:
            print('EPISODE END!')
            print('Total Frame Number: ' + str((loop_count_15hz + 1)*4))
            print('Episode Number: ' + str(episode_no))
            print('Episode Score: ' + str(episode_reward))
            print(' ')
            episode_reward = 0
            episode_no += 1
    
    np.savetxt(os.path.join(args.save_path, 'reward_vec_' + args.session_no + '.csv'), episode_reward_vec_15hz, delimiter=",", fmt='%01.1f')
    np.savetxt(os.path.join(args.save_path, 'episode_vec_' + args.session_no + '.csv'), episode_no_vec_15hz, delimiter=",", fmt='%01.0u')
    
    print('Session ' + str(args.session_no) + ' completed!')
    print('Total session score: ' + str(session_reward))
    print(' ')

if __name__ == '__main__':
    main()
