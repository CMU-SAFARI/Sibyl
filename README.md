# Sibyl (Software implementations will be available soon!)

Sibyl is the first technique that uses reinforcement learning for data placement in hybrid storage systems. Sibyl observes different features of the running workload as well as the storage devices to make system-aware data placement decisions. For every decision it makes, Sibyl receives a reward from the system that it uses to evaluate the long-term performance impact of its decision and continuously optimizes its data placement policy online.

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
## Prerequisites
* tensorflow 2.4.1
* tf-agents 0.7.1
* tensorflow-probability 0.12.2
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

## Evaluation Setup
We evaluate Sibyl using real systems with various hybrid storage system (HSS) configurations. The HSS devices appear as a single flat block device that exposes one contiguous logical block address space to the OS. We use the [TF-Agents API](https://github.com/tensorflow/agents) to develop Sibyl. We run the Linux Mint 20.1 operating system with the  Ext3 file system.

## Contact
Gagandeep Singh (gagsingh at ethz dot com)

## Acknowledgements
We thank the SAFARI Research Group members for valuable feedback and the stimulating intellectual environment they provide. We acknowledge the generous gifts of our industrial partners, especially Google, Huawei, Intel, Microsoft, VMware. This research was partially supported by the Semiconductor Research Corporation and the ETH Future Computing Laboratory.

