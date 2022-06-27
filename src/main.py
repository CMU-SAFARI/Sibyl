#!/usr/bin/env python
import tensorflow as tf
from tensorflow.python.client import device_lib
from tf_agents.experimental.train.utils import spec_utils
from tf_agents.experimental.train.utils import strategy_utils
from tf_agents.experimental.train.utils import train_utils
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.policies import random_tf_policy
from tf_agents.metrics import tf_py_metric
from tf_agents.metrics import tf_metrics
from tf_agents.metrics import py_metric
from tf_agents.drivers import py_driver
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.networks import q_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.replay_buffers import py_hashed_replay_buffer
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.eval import metric_utils
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import random_tf_policy
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.policies import policy_saver
from enum import Enum
import os
from ctypes import *
import sys
#Environment
from hybridstorage import HybridStorage
from hybridstorageenvironment import HybridStorageEnvironment
import functools
from tensorflow.keras import datasets, layers, models
from tf_agents.networks import sequential
import time, array, random, copy, math

trace='hm_1'
traceLength= sum(1 for line in open(trace+".csv"))

#Setting hyperparameters
algo='C51' # DQN, DDQN, PPO, REINFORCE
batch_size =64 #64,256  # @param {type:"integer"}
learning_rate = 2.5e-4 #1e-3, 2.5e-4 # @param {type:"number"}
replay_buffer_capacity=10000#<r_buff> #2000 , 100000,50000
initial_collect_steps = 10000  # @param {type:"integer"} 
collect_steps_per_iteration = 2  # @param {type:"integer"} update_period=4
collect_episodes=2
eps=1
gam=0.9
KERAS_LSTM_FUSED = 2
num_iterations=100
eval_interval =25  # @param {type:"integer"}
fc_layer_params =(75,50) 
log_interval = 1 # @param {type:"integer"}
num_eval_episodes = 1 # @param {type:"integer"}
lstm_size=(20,)
num_atoms = 51  # @param {type:"integer"}
min_q_value = -20  # @param {type:"integer"}
max_q_value = 20  # @param {type:"integer"}
n_step_update = 2  # @param {type:"integer"}
orig_stdout = sys.stdout
sys.stdout = open("log.txt",'wt')
memEnvironemt = HybridStorageEnvironment(HybridStorage(trace))
print('Observation Spec:')
print(memEnvironemt.time_step_spec().observation)
print('Action Spec:')
print(memEnvironemt.action_spec())
print('Reward Spec:')
print(memEnvironemt.time_step_spec().reward)

testEnvironemt = HybridStorageEnvironment(HybridStorage(trace))

num_parallel_environments=6

train_env = tf_py_environment.TFPyEnvironment(memEnvironemt)
eval_env = tf_py_environment.TFPyEnvironment(testEnvironemt)

all_train_loss = []
all_metrics = []
returns=[]

def compute_avg_return(environment, policy, num_episodes):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0
        i=0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
            i +=1
        total_return += episode_return
    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

def create_recurrent_network(
    input_fc_layer_units,
    lstm_size,
    output_fc_layer_units,
    num_actions):
  rnn_cell = tf.keras.layers.StackedRNNCells(
      [fused_lstm_cell(s) for s in lstm_size])
  return models.Sequential(
      [dense(num_units) for num_units in input_fc_layer_units]
      + [dynamic_unroll_layer.DynamicUnroll(rnn_cell)]
      + [dense(num_units) for num_units in output_fc_layer_units]
      + [logits(num_actions)])  
def create_feedforward_network(fc_layer_units, num_actions):
  return sequential.Sequential(
      [dense(num_units) for num_units in fc_layer_units]
      + [logits(num_actions)])


random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                              train_env.action_spec())

logits = functools.partial(
    tf.keras.layers.Dense,
    activation=None,
    kernel_initializer=tf.compat.v1.initializers.random_uniform(
        minval=-0.03, maxval=0.03),
    bias_initializer=tf.compat.v1.initializers.constant(-0.2))

fused_lstm_cell = functools.partial(
    tf.keras.layers.LSTMCell, implementation=KERAS_LSTM_FUSED)
dense = functools.partial(
    tf.keras.layers.Dense,
    activation=tf.keras.activations.relu,
    kernel_initializer=tf.compat.v1.variance_scaling_initializer(
        scale=2.0, mode='fan_in', distribution='truncated_normal'))

num_actions = train_env.action_spec().maximum - train_env.action_spec().minimum + 1
action_spec = train_env.action_spec()
num_actions = action_spec.maximum - action_spec.minimum + 1
train_step_counter = tf.Variable(0)
update_period = 4

optimizer = tf.keras.optimizers.RMSprop ( lr = 2.5e-4 , rho = 0.95 , momentum = 0.0,
                                epsilon = 0.00001 , centered = True )

epsilon_fn = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=1.0,  # initial ε

                decay_steps=traceLength // update_period,
                end_learning_rate=0.01) # final ε


if(algo=='DQN'):
    from tf_agents.keras_layers import dynamic_unroll_layer
    #with strategy.scope():
    def create_recurrent_network(
        input_fc_layer_units,
        lstm_size,
        output_fc_layer_units,
        num_actions):
      rnn_cell = tf.keras.layers.StackedRNNCells(
          [fused_lstm_cell(s) for s in lstm_size])
      return sequential.Sequential(
          [dense(num_units) for num_units in input_fc_layer_units]
          + [dynamic_unroll_layer.DynamicUnroll(rnn_cell)]
          + [dense(num_units) for num_units in output_fc_layer_units]
          + [logits(num_actions)])
    from tf_agents.networks import sequential
    q_net = q_network.QNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=fc_layer_params)

    q_net = q_network.QNetwork(
            train_env.observation_spec(),
            train_env.action_spec(),
            fc_layer_params=fc_layer_params,
            activation_fn=tf.nn.swish)
    tf_agent = dqn_agent.DqnAgent(
            train_env.time_step_spec(),
            train_env.action_spec(),
            q_network=q_net,
            optimizer=optimizer,
            target_update_period=2000,
            td_errors_loss_fn=tf.keras.losses.Huber(reduction="none"),
            gamma=gam, # discount factor
            debug_summaries=True,
          summarize_grads_and_vars=True,
            train_step_counter=train_step_counter,
            epsilon_greedy=lambda: epsilon_fn(train_step_counter))
if(algo=='DDQN'):
    #with strategy.scope():
    q_net = q_network.QNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=fc_layer_params)

    q_net = q_network.QNetwork(
            train_env.observation_spec(),
            train_env.action_spec(),
            fc_layer_params=fc_layer_params,
            activation_fn=tf.nn.swish)
    tf_agent = dqn_agent.DdqnAgent(
            train_env.time_step_spec(),
            train_env.action_spec(),
            q_network=q_net,
            optimizer=optimizer,
            target_update_period=2000,
            debug_summaries=True,
          summarize_grads_and_vars=True,
            td_errors_loss_fn=tf.keras.losses.Huber(reduction="none"),
            gamma=gam, # discount factor
            train_step_counter=train_step_counter,
            epsilon_greedy=lambda: epsilon_fn(train_step_counter))

if(algo=='C51'):
    #with strategy.scope():
        from tf_agents.networks import categorical_q_network
        from tf_agents.agents.categorical_dqn import categorical_dqn_agent
        train_step_counter = tf.Variable(0)
        optimizer = tf.keras.optimizers.RMSprop(lr=learning_rate, rho=0.95, momentum=0.0, epsilon=0.00001, centered=True)
        epsilon_fn = tf.keras.optimizers.schedules.PolynomialDecay(
                        initial_learning_rate=1.0, 
                        decay_steps=traceLength // update_period,
                        end_learning_rate=0.0001,
                        power=0.6)
        categorical_q_net = categorical_q_network.CategoricalQNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        activation_fn=tf.nn.swish,
        num_atoms=num_atoms,
        fc_layer_params=fc_layer_params)
        tf_agent = categorical_dqn_agent.CategoricalDqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        categorical_q_network=categorical_q_net,
        epsilon_greedy=lambda: epsilon_fn(train_step_counter),
        #epsilon_greedy=<epsilon_val>,
        optimizer=optimizer,
        min_q_value=min_q_value,
        max_q_value=max_q_value,
        n_step_update=n_step_update,
        # td_errors_loss_fn=tf.keras.losses.Huber(reduction="none"),
        td_errors_loss_fn=common.element_wise_squared_loss,
        gamma=gam,
          debug_summaries=True,
          summarize_grads_and_vars=True,
        train_step_counter=train_step_counter)

if (algo=='REINFORCE'):
    from tf_agents.networks import value_network
    use_value_network=True
    #with strategy.scope():
    actor_net = actor_distribution_network.ActorDistributionNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    activation_fn=tf.nn.swish,
    fc_layer_params=fc_layer_params)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    train_step_counter = tf.compat.v2.Variable(0)

    if use_value_network:
      value_net = value_network.ValueNetwork(
          train_env.time_step_spec().observation,
          fc_layer_params=fc_layer_params)

    tf_agent = reinforce_agent.ReinforceAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        actor_network=actor_net,
        value_network=value_net if use_value_network else None,
        optimizer=optimizer,
        gamma=gam,
        normalize_returns=True,
        train_step_counter=train_step_counter)

if(algo=='PPO'):
    from tf_agents.agents.ppo import ppo_clip_agent
    from tf_agents.agents.ppo import ppo_agent
    from tf_agents.drivers import dynamic_episode_driver
    from tf_agents.environments import parallel_py_environment
    from tf_agents.environments import suite_mujoco
    from tf_agents.eval import metric_utils
    from tf_agents.metrics import tf_metrics
    from tf_agents.networks import actor_distribution_network
    from tf_agents.networks import actor_distribution_rnn_network
    from tf_agents.networks import value_network
    from tf_agents.networks import value_rnn_network

    num_parallel_environments=30
    num_epochs=25
 
    summaries_flush_secs=1
    use_tf_functions=True
    debug_summaries=True
    summarize_grads_and_vars=True

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    use_rnns=False
    train_step_counter = tf.compat.v1.train.get_or_create_global_step()
    if use_rnns:
      actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
          train_env.observation_spec(),
          train_env.action_spec(),
          activation_fn=tf.nn.swish,
          input_fc_layer_params=fc_layer_params,
          output_fc_layer_params=None)
      value_net = value_rnn_network.ValueRnnNetwork(
          train_env.observation_spec(),
          input_fc_layer_params=fc_layer_params,
          output_fc_layer_params=None)
    else:
      actor_net = actor_distribution_network.ActorDistributionNetwork(
          train_env.observation_spec(),
          train_env.action_spec(),
          activation_fn=tf.nn.swish,
          fc_layer_params=fc_layer_params
          )
      value_net = value_network.ValueNetwork(
          train_env.observation_spec(),
          fc_layer_params=fc_layer_params,
          activation_fn=tf.nn.swish)

    tf_agent = ppo_clip_agent.PPOClipAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        optimizer,
        actor_net=actor_net,
        value_net=value_net,
        entropy_regularization=0.0,
        importance_ratio_clipping=0.2,
        normalize_observations=False,
        normalize_rewards=False,
        use_gae=True,
        num_epochs=num_epochs,
        debug_summaries=debug_summaries,
        
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=train_step_counter)

tf_agent.initialize()
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=tf_agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_capacity)

def collect_data(environment, policy, buffer,steps,metrics=None):
    if(metrics!=None):
            observe=[buffer.add_batch]+metrics
    else:
        observe=[buffer.add_batch]
    dynamic_step_driver.DynamicStepDriver(environment,policy, observers=observe, num_steps=steps).run()

def collect_episode(environment, policy, buffer,episodes,metrics=None):
    if(metrics!=None):
            observe=[buffer.add_batch]+metrics
    else:
        observe=[buffer.add_batch]
    dynamic_episode_driver.DynamicEpisodeDriver(environment,policy, observers=observe, num_episodes=episodes).run()


if(algo in[ 'DDQN','TD3','DQN']):

    common.function(collect_data(train_env, random_policy, replay_buffer,initial_collect_steps))
    # Dataset generates trajectories with shape [Bx2x...]
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, 
        sample_batch_size=batch_size, 
        num_steps=2).prefetch(64)
    iterator = iter(dataset)


if algo in ['C51']:
  
    common.function(collect_data(train_env, random_policy, replay_buffer,initial_collect_steps))
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, 
        sample_batch_size=batch_size, 
        num_steps=n_step_update +1).prefetch(16)
 
    iterator = iter(dataset)

tf_agent.train = common.function(tf_agent.train)

# Reset the train step
tf_agent.train_step_counter.assign(0)
train_metrics = [tf_metrics.MaxReturnMetric(),tf_metrics.ChosenActionHistogram(),tf_metrics.AverageReturnMetric(), tf_metrics.AverageEpisodeLengthMetric()]

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
returns = [avg_return]
total_loss=[]
all_train_loss = []
all_metrics = []
for it in range(num_iterations):
 
    current_metrics = []
    # Collect a few episodes using collect_policy and save to the replay buffer.
    if (algo in ['REINFORCE','PPO']):
        common.function(collect_episode(train_env, tf_agent.collect_policy,replay_buffer, collect_episodes,metrics=train_metrics))
        experience = replay_buffer.gather_all()
        train_loss_in = tf_agent.train(experience)
        replay_buffer.clear()

    if algo in ['C51','DDQN','DQN']:
        common.function(collect_data(train_env, tf_agent.collect_policy, replay_buffer,collect_steps_per_iteration,metrics=train_metrics))
        experience, unused_info = next(iterator)
        train_loss_in = tf_agent.train(experience)

    step = tf_agent.train_step_counter.numpy()
    for i in range(len(train_metrics)):
            current_metrics.append(train_metrics[i].result().numpy())
    all_metrics.append(current_metrics)
    # print(all_metrics)
    if step % log_interval == 0:
        total_loss.append(train_loss_in.loss)
        print('step = {0}: loss = {1}'.format(step, train_loss_in.loss))

    if step % eval_interval == 0:
        avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
        returns.append(avg_return)

#Saving the agent
tf.saved_model.save(tf_agent, "agent")
#Saving the policy
agent_policy = tf_agent.policy
policy_saver.PolicySaver(agent_policy).save("sibyl_policy")

for i in range(len(train_metrics)):
                print('{}: {}'.format(train_metrics[i].name, train_metrics[i].result().numpy()))

# Reset the train step
tf_agent.train_step_counter.assign(0)

#reset eval environment
eval_env.reset()

###########ends###############
sys.stdout.close()
sys.stdout=orig_stdout 



