# Sibyl

Hybrid storage systems (HSS) use multiple different storage devices to provide high and scalable storage capacity at high performance. Data placement across different devices is critical to maximize the benefits of such a hybrid system. 

Sibyl is the first technique that uses reinforcement learning for data placement in hybrid storage systems. Sibyl observes different features of the running workload as well as the storage devices to make system-aware data placement decisions. For every decision it makes, Sibyl receives a reward from the system that it uses to evaluate the long-term performance impact of its decision and continuously optimizes its data placement policy online.


## About The Framework
We evaluate Sibyl using real systems with various HSS configurations. The HSS devices appear as a single flat block device that exposes one contiguous logical block address space to the OS. We use the [TF-Agents](https://github.com/tensorflow/agents) to develop Sibyl. We run the Linux Mint 20.1 operating system with the  Ext3 file system.

## Prerequisites
* tensorflow 2.4.1
* tf-agents 0.7.1
* pandas 1.1.3

## Installation
```
$ git clone https://github.com/CMU-SAFARI/Sibyl.git # or fork first and clone that
$ cd Sibyl
$ python3 -m venv venv3
$ source venv3/bin/activate
(venv3) $ pip install -r requirements.txt
(venv3) $ python setup.py develop
```

## Preparing HSS Drivers

## Running
```
sibyl execute workload_path driver_path
```

`workload_path`: Path to a workload's storage trace

`driver_path`: Path to hybrid storage system driver

Additional options: 

`--rl_algo`: Choose different RL algorithms. Current options include C51, DQN, DDQN, REINFORCE, and PPO

`--batch`: The number of
samples processed in each training iteration

`--lr`:  Determines the rate at
which neural network weights are updated

`--buf_cap`: Experience buffer size

`--eps`: The exploration rate (𝜖) balances exploration and exploitation rate

`--gam`: The discount factor (𝛾) determines the balance between the immediate and future
rewards

`--num_itr`: The number of times to run an environment

`--eval_itr`: When to evaluate the learned policy

Saving agent and policy:

```
#Saving the agent
tf.saved_model.save(tf_agent, "agent")
#Saving the policy
agent_policy = tf_agent.policy
policy_saver.PolicySaver(agent_policy).save("sibyl_policy")
```

Load saved policy:

```
saved_policy = tf.saved_model.load(sibyl_policy)
```
## Citation
>Gagandeep Singh, Rakesh Nadig, Jisung Park, Rahul Bera, Nastaran Hajinazar, David Novo; Juan Gomez-Luna, Sander Stuijk, Henk Corporaal, Onur Mutlu.
[**"Sibyl: Adaptive and Extensible Data Placement in Hybrid Storage Systems using Online Reinforcement Learning"**](https://people.inf.ethz.ch/omutlu/pub/Sibyl_RL-based-data-placement-in-hybrid-storage-systems_isca22.pdf)
In _Proceedings of the 49th International Symposium on Computer Architecture (ISCA),_ New York City, NY, USA, June 2022.

Bibtex entry for citation:

```
@inproceedings{singh2022sibyl,
  title={Sibyl: Adaptive and Extensible Data Placement in Hybrid Storage Systems Using Online Reinforcement Learning},
  author={Singh, Gagandeep and Nadig, Rakesh and Park, Jisung and Bera, Rahul and Hajinazar, Nastaran and Novo, David and G{\'o}mez-Luna, Juan and Stuijk, Sander and Corporaal, Henk and Mutlu, Onur},
  booktitle={ISCA},
  year={2022},
}
```

Please also cite [TFAgents](https://github.com/tensorflow/agents):

```
@misc{TFAgents,
  title = { {TF-Agents}: A library for Reinforcement Learning in TensorFlow},
  author = {Sergio Guadarrama and Anoop Korattikara and Oscar Ramirez and
     Pablo Castro and Ethan Holly and Sam Fishman and Ke Wang and
     Ekaterina Gonina and Neal Wu and Efi Kokiopoulou and Luciano Sbaiz and
     Jamie Smith and Gábor Bartók and Jesse Berent and Chris Harris and
     Vincent Vanhoucke and Eugene Brevdo},
  howpublished = {\url{https://github.com/tensorflow/agents} },
  url = "https://github.com/tensorflow/agents",
  year = 2018,
  note = "[Online; accessed 25-June-2019]"
}
```
## Contact
Gagandeep Singh (gagsingh at ethz dot com)

## Acknowledgements
We thank the SAFARI Research Group members for valuable feedback and the stimulating intellectual environment they provide. We acknowledge the generous gifts of our industrial partners.

