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
from tf_agents.agents.dqn import dqn_agent
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
from sibyl.src.hybridstorage import HybridStorage
from sibyl.src.hybridstorageenvironment import HybridStorageEnvironment
from sibyl.src.trihybridstorage import TriHybridStorage
from sibyl.src.trihybridstorageenvironment import TriHybridStorageEnvironment

from sibyl.src.utils import compute_avg_return, create_recurrent_network, create_feedforward_network
import functools
from tensorflow.keras import datasets, layers, models
from tf_agents.networks import sequential
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.drivers import dynamic_step_driver
import time, array, random, copy, math
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from tqdm import trange
import logging
logging.basicConfig(filename='rl_debug.log', filemode='w', format='%(asctime)s %(name)s - %(levelname)s - %(message)s',level=logging.INFO, datefmt='%m/%d/%Y %I:%M:%S %p')

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

def main(args):
    trace=args.workload_path
    # trace = os.path.expanduser(args.workload_path)
    logging.info("Trace=%s" %trace)
    if not os.path.isfile(trace):
        print("[error] %s cannot read" % trace)
        exit(1)
    traceLength= sum(1 for line in open(trace))

    #Setting hyperparameters
    algo =args.rl_algo # DQN, DDQN, PPO, REINFORCE
    logging.info("RL algo=%s" %algo)
    batch_size = args.batch #64,256  # @param {type:"integer"}
    learning_rate = args.lr #1e-3, 2.5e-4 # @param {type:"number"}
    replay_buffer_capacity = args.buf_cap #<r_buff> #2000 , 100000,50000
    initial_collect_steps = args.init_collect  # @param {type:"integer"} 
    collect_steps_per_iteration = 2  # @param {type:"integer"} update_period=4
    collect_episodes = 2
    eps = args.eps
    gam = args.gam
    num_iterations = args.num_itr
    eval_interval = args.eval_itr  # @param {type:"integer"}
    fc_layer_params = (20,30) 
    log_interval = 1 # @param {type:"integer"}
    num_eval_episodes = 1 # @param {type:"integer"}
    num_atoms = 51  # @param {type:"integer"}
    min_q_value = -20  # @param {type:"integer"}
    max_q_value = 20  # @param {type:"integer"}
    n_step_update = 2  # @param {type:"integer"}

    orig_stdout = sys.stdout
    so_path=args.so_path
    type_env=args.type_env
    if(type_env=="dual"):
        memEnvironemt = HybridStorageEnvironment(HybridStorage(trace,so_path))
        testEnvironemt = HybridStorageEnvironment(HybridStorage(trace,so_path))
    elif(type_env=="tri"):  
        memEnvironemt = TriHybridStorageEnvironment(TriHybridStorage(trace,so_path))
        testEnvironemt = TriHybridStorageEnvironment(TriHybridStorage(trace,so_path))
    else:
        logging.error("Unsupported type %s" % type_env)
        exit(1)
    logging.info("Observation Spec={}".format(memEnvironemt.time_step_spec().observation))
    logging.info("Action Spec={}".format(memEnvironemt.action_spec()))
    logging.info("Reward Spec={}".format(memEnvironemt.time_step_spec().reward))

    

    num_parallel_environments=6

    train_env = tf_py_environment.TFPyEnvironment(memEnvironemt)
    eval_env = tf_py_environment.TFPyEnvironment(testEnvironemt)

    all_train_loss = []
    all_metrics = []
    returns=[]

    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                  train_env.action_spec())

    logits = functools.partial(
        tf.keras.layers.Dense,
        activation=None,
        kernel_initializer=tf.compat.v1.initializers.random_uniform(
            minval=-0.03, maxval=0.03),
        bias_initializer=tf.compat.v1.initializers.constant(-0.2))


    num_actions = train_env.action_spec().maximum - train_env.action_spec().minimum + 1
    action_spec = train_env.action_spec()
    num_actions = action_spec.maximum - action_spec.minimum + 1
    train_step_counter = tf.Variable(0)
    update_period = 4

    optimizer = tf.keras.optimizers.RMSprop ( lr = 2.5e-4 , rho = 0.95 , momentum = 0.0,
                                    epsilon = eps , centered = True )

    epsilon_fn = tf.keras.optimizers.schedules.PolynomialDecay(
                    initial_learning_rate=1.0,  # initial ε
                    decay_steps=traceLength // update_period,
                    end_learning_rate=0.01) # final ε


    if(algo=='DQN'):
        from tf_agents.keras_layers import dynamic_unroll_layer
        #with strategy.scope():
       
        from tf_agents.networks import sequential

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
            optimizer = tf.keras.optimizers.RMSprop(lr=learning_rate, rho=0.95, momentum=0.0, epsilon=eps, centered=True)

            categorical_q_net = categorical_q_network.CategoricalQNetwork(
                train_env.observation_spec(),
                train_env.action_spec(),
                activation_fn=tf.nn.swish,
                num_atoms=num_atoms,
                fc_layer_params=fc_layer_params)
            target_categorical_q_net = categorical_q_network.CategoricalQNetwork(
                train_env.observation_spec(),
                train_env.action_spec(),
                activation_fn=tf.nn.swish,
                num_atoms=num_atoms,
                fc_layer_params=fc_layer_params)
            tf_agent = categorical_dqn_agent.CategoricalDqnAgent(
                train_env.time_step_spec(),
                train_env.action_spec(),
                categorical_q_network=categorical_q_net,
                target_categorical_q_network=target_categorical_q_net,
                epsilon_greedy=lambda: epsilon_fn(train_step_counter),
                optimizer=optimizer,
                min_q_value=min_q_value,
                max_q_value=max_q_value,
                n_step_update=n_step_update,
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
            num_steps=n_step_update +1).prefetch(64)
     
        iterator = iter(dataset)

    tf_agent.train = common.function(tf_agent.train)

    # Reset the train step
    tf_agent.train_step_counter.assign(0)
    num_episodes = tf_metrics.NumberOfEpisodes()
    env_steps = tf_metrics.EnvironmentSteps()   
    train_metrics = [num_episodes,env_steps,tf_metrics.MaxReturnMetric(),tf_metrics.ChosenActionHistogram(),tf_metrics.AverageReturnMetric(), tf_metrics.AverageEpisodeLengthMetric()]

    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
    returns = [avg_return]
    total_loss=[]
    all_train_loss = []
    all_metrics = []


    current_metrics = []
    
    for _ in trange(num_iterations, desc="Episode", unit="episodes"):
     
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
        if step % log_interval == 0:
            total_loss.append(train_loss_in.loss)
            logging.info('step = {}: loss = {}'.format(step, train_loss_in.loss))

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
            logging.info('avg_return = {}'.format(avg_return))
            returns.append(avg_return)

    #Saving the agent
    tf.saved_model.save(tf_agent, "agent")
    #Saving the policy
    agent_policy = tf_agent.policy
    policy_saver.PolicySaver(agent_policy).save("sibyl_policy")
    print('Number of Steps: ', env_steps.result().numpy())
    print('Number of Episodes: ', num_episodes.result().numpy())
    for i in range(len(train_metrics)):
                    print('{}: {}'.format(train_metrics[i].name, train_metrics[i].result().numpy()))
    print("Returns for evaluated episodes:",returns)
    print("Training loss:",total_loss)
    # Reset the train step
    tf_agent.train_step_counter.assign(0)

    #reset eval environment
    eval_env.reset()
    logging.info("> Ending RL Run\n")
    ###########ends###############
    

def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("type_env",default="dual")
    parser.add_argument("workload_path")
    parser.add_argument("so_path")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--rl_algo', default="DQN")
    parser.add_argument("--batch", default=128, type=int)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--buf_cap", default=1000, type=int)
    parser.add_argument("--init_collect", default=1000, type=int)
    parser.add_argument("--eps", default=1, type=float)
    parser.add_argument("--gam", default=0.99, type=float)
    parser.add_argument("--num_itr", default=1000, type=int)
    parser.add_argument("--eval_itr", default=1, type=int)
    return parser
if __name__ == "__main__":
    main()