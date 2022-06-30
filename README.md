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
The storage management layer in the OS orchestrates host I/O requests across heterogeneous devices, which are connected via an NVM Express (NVMe) or SATA interface. The storage management layer provides the operating system with a unified logical address space (like the multiple device driver (md) kernel module in Linux). The unified logical address space is divided into a number of logical pages (e.g., 4 KiB pages). The pages in the logical address space are assigned to an application. The storage management layer translates a read/write performed on a logical page into a read/write operation on a target storage device based on the data placement policy. In addition, the storage management layer manages data migration between the storage devices in an HSS. When data currently stored in the slow storage device is moved to the fast storage device, it is called promotion. Promotion is usually performed when a page in the slow storage device is accessed frequently. Data is moved from the fast storage device to the slow storage device during an eviction. Eviction typi- cally occurs when the data in the fast storage device is infrequently accessed or when the fast storage device becomes full.

Compiling HSS drivers:

```
$ cd drivers
$ gcc -fPIC -shared -o Sibyl_lib.so Sibyl_lib.c
```

## Running
```
sibyl execute workload_path driver_path
```

`workload_path`: Path to a workload's storage trace

`driver_path`: Path to hybrid storage system driver

Additional options: 

`--rl_algo`: Choose a suitable RL algorithm for your environment. Current options include C51, DQN, DDQN, REINFORCE, and PPO

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

Rakesh Nadig (nadigr at ethz dot com)

## Acknowledgements
We thank the SAFARI Research Group members for valuable feedback and the stimulating intellectual environment they provide. We acknowledge the generous gifts of our industrial partners.

