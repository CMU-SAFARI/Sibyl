#!/usr/bin/env python
import tensorflow as tf

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

