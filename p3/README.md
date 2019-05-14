[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: MADDPG_learning_curve.png "MADDPG learning curve"
[image3]: MA4DPG_learning_curve.png "MA4DPG learning curve"


# Project 3: Collaboration and Competition

### Introduction

For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

* Set-up: Two-player game where agents control rackets to bounce ball over a
  net.
* Goal: The agents must bounce ball between one another while not dropping or
  sending ball out of bounds. 
* Agents: The environment contains two agent linked to a single Brain named
  TennisBrain. 
* Agent Reward Function (independent):
  * +0.1 To agent when hitting ball over net.
  * -0.01 To agent who let ball hit their ground, or hit ball out of bounds.
* Brains: One Brain with the following observation/action space.
  * Vector Observation space: 8 variables corresponding to position and velocity
    of ball and racket.
  * Vector Action space: (Continuous) Size of 2, corresponding to movement
    toward net or away from net, and jumping.
* Solving environment: The environment is considered solved, when the average (over 100 episodes) of the **scores** is at least +0.5. We take the maximum of 2 scores (of each agents).


### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

2. Place the file in the DRLND GitHub repository, in the `p3/` folder, and unzip (or decompress) the file. 

3. Follow the instructions in `Tennis.ipynb` to get started with training an agent with the DDPG algorithm! Or use `Tennis_D4PG.ipynb` to do it with the D4PG algorithm. 

### Benchmark

- DDPG (Deep Deterministic Policy Gradient) is an actor-critic algorithm with two seperate networks, one for taking actions and the other for estimating the V or Q-values. 
- D4PG (Distributed Distributional) applies a set of improvements over DDPG, among one of them being the estimated Q-values as a random variable. 
- MADDPG (Multi-Agent) are multiple DDPG networks designed for cooperative or competitative games played by multiple agents. 

### Results

As can seen from the plot below, the environment was solved in 5594 episodes by the DDPG algorithm! But look at the next image: The D4PG algorithm saved the environment only in 477 episodes, more than 10 times faster! 
![MADDPG learning curve][image2]

![MA4DPG learning curve][image3]

### Model details

#### DDPG
The weights of the critic are updated to minimize the difference between the predicted and actual Q-values:

```math
SE = \frac{\sigma}{\sqrt{n}}
```

In code we do this by by minimizing the difference between the TD target (Q(st+1,at+1) from the target networks) and the expected values (Q(st,at) from the critic network):
```python
q_targets = rewards + (gamma * q_targets_next))
q_expected = critic(states, actions)
critic_loss = F.mse_loss(q_expected, q_targets)
```

The weights of the actor are updated to maximize the reward, estimated from the critic:
```python
actor_loss = -self.critic_local(states, actions_pred).mean()
```

#### D4PG







