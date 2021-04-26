#!/usr/bin/env python
# coding: utf-8

# In[18]:


import torch
import random
import gym
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque


# In[19]:


class policynetwork(nn.Module):
    def __init__(self,S,L,A):
        super(policynetwork,self).__init__()
        self.l=nn.Linear(S,L)
        self.o=nn.Linear(L,A)
            
        def forward(self,x):
            x=F.relu(self.l(x))
            x=F.softmax(self.o(x))


# In[ ]:


env=gym.make("CartPole-v1")
policy=policynetwork(env.observation_space.shape[0],20,env.action_space.n)
optimizer=torch.optim.Adam(policy.parameters())
n_episode=1000
gamma=0.99
returns=deque(maxlen=100)
render_rate=100



while True:
    rewards=[]
    actions=[]
    states=[]
    state=env.reset()
    while True:
        env.render()
        probs=policy(torch.tensor(state).unsqueeze(0).float())
        sampler=Categorical(probs)
        action=sampler.sample()
        new_state,reward,done,info=env.step(action.item())
        states.append(state)
        rewards.append(reward)
        actions.append(action)
        state=new_state
        if done:
            break
    
        rewards=np.array(rewards)
        R=torch.sum(rewards)
    
        states=torch.tensor(states).float()
        actions=torch.tensor(actions)
        probs = policy(states)
        sampler = Categorical(probs)
        log_probs = -sampler.log_prob(actions)   # "-" because it was built to work with gradient descent, but we are using gradient ascent
        pseudo_loss = torch.sum(log_probs * R) # loss that when differentiated with autograd gives the gradient of J(θ)
    # upd1ate policy weights
        optimizer.zero_grad()
        pseudo_loss.backward()
        optimizer.step()

    # calculate average return and print it out
        returns.append(np.sum(rewards))
        print("Episode: {:6d}\tAvg. Return: {:6.2f}".format(n_episode, np.mean(returns)))
        n_episode += 1
env.close()


# In[ ]:




