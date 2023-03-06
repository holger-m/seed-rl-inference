from seed_rl.atari import networks
from seed_rl.common import utils
import math
import tensorflow as tf
import numpy as np
import cv2
import gym

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


#firstnet = networks.DuelingLSTMDQNNet(18, (84, 84, 1), 4)

#firstnet.load_weights('/workspace/container_mount/data/Checkpoints_from_google/SpaceInvaders/0/ckpt-130')


def create_agent_fn():
    return networks.DuelingLSTMDQNNet(18, (84, 84, 1), 4)
  
agent = create_agent_fn()
target_agent = create_agent_fn()

def create_optimizer_fn(unused_final_iteration):
    learning_rate_fn = lambda iteration: 0.00048
    optimizer = tf.keras.optimizers.Adam(0.00048, epsilon=1e-3)
    return optimizer, learning_rate_fn
    
def get_replay_insertion_batch_size():
    return int(64 / 1.5)

iter_frame_ratio = (get_replay_insertion_batch_size() * 100 * 1)
final_iteration = int(math.ceil(int(1e9) / iter_frame_ratio))
optimizer, learning_rate_fn = create_optimizer_fn(final_iteration)

ckpt = tf.train.Checkpoint(agent=agent, target_agent=target_agent, optimizer=optimizer)
ckpt.restore('/workspace/container_mount/data/Checkpoints_from_google/SpaceInvaders/0/ckpt-130')



env = gym.make('SpaceInvadersNoFrameskip-v4', full_action_space=True)
env.seed(0)  # hard-coded in learner.py line 503: env = create_env_fn(0, FLAGS)

env.reset()

env.step(0)  # no action

obs_dims = env.observation_space

screen_buffer = np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8)



env.ale.getScreenGrayscale(screen_buffer)

def _pool_and_resize(screen_buffer):
    transformed_image = cv2.resize(screen_buffer, (84, 84), interpolation=cv2.INTER_LINEAR)
    int_image = np.asarray(transformed_image, dtype=np.uint8)
    return np.expand_dims(int_image, axis=2)

#observation = _pool_and_resize(screen_buffer)

#env.observation_space.shape = (84, 84, 1)

env_output_specs = utils.EnvOutput(
      tf.TensorSpec([], tf.float32, 'reward'),
      tf.TensorSpec([], tf.bool, 'done'),
      tf.TensorSpec(env.observation_space.shape, env.observation_space.dtype,
                    'observation'),
      tf.TensorSpec([], tf.bool, 'abandoned'),
      tf.TensorSpec([], tf.int32, 'episode_step'),
)
action_specs = tf.TensorSpec([], tf.int32, 'action')
agent_input_specs = (action_specs, env_output_specs)

initial_agent_state = agent.initial_state(1)
agent_state_specs = tf.nest.map_structure(lambda t: tf.TensorSpec(t.shape[1:], t.dtype), initial_agent_state)
input_ = tf.nest.map_structure(lambda s: tf.zeros([1] + list(s.shape), s.dtype), agent_input_specs)

encode = lambda x: x
decode = lambda x, s=None: x if s is None else tf.nest.pack_sequence_as(s, x)



input_ = encode(input_)
input_action, input_env = input_

input_env = input_env._replace(observation=tf.convert_to_tensor(np.ones((1, 210, 160, 1), dtype=np.uint8)))
input_ = (input_action, input_env)


#input_env_reward, input_env_done, input_env_observation, input_env_abandoned, input_env_episode_step = input_env

#input_env_observation = tf.convert_to_tensor(np.ones((1, 84, 84, 1), dtype=np.uint8))

#print(type(input_env_observation))
#print(len(input_env_observation))
#print(input_env_observation)

#input_env = (input_env_reward, input_env_done, input_env_observation, input_env_abandoned, input_env_episode_step)
#input_ = (input_action, input_env)

# from networks.py:
#prev_actions, env_outputs = input_
#stacked_frames, frame_state = stack_frames(observation, agent_state.frame_stacking_state, done, self._stack_size)
#env_outputs = env_outputs._replace(observation=stacked_frames / 255)

#input_[1][2] = np.ones((1, 84, 84, 1), dtype=np.uint8)

#print(env.observation_space.shape)
#print(input_[1][2][0,:,:,0])
#print(type(input_))
print(input_)

rand_net_out = agent(input_, initial_agent_state)

#print('rand_net_out:')
#print(rand_net_out)

# continue with learner.py line 770 and encode/decode from utils.py lines 104, 105

