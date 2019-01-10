#!/usr/bin/env python
import sys, os, time, copy, random
import pickle, json
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
# Using Pytorch
import torch
from torch import nn, optim
from torch.autograd.variable import Variable as V
from torchvision import models, utils

def load_rollout(filename):
    with open(filename, 'rb') as f:
        data=pickle.loads(f.read())
    return data

class simple_neuralnet(torch.nn.Module):
    def __init__(self,input_shape,output_shape):
        super(simple_neuralnet,self).__init__()
        self.bn1=nn.Sequential(
            nn.BatchNorm1d(input_shape)
        )
        self.fc1=nn.Sequential(
            nn.Linear(input_shape,128)
            nn.LeakyReLU(0.2)
        )
        self.output=nn.Sequential(
            nn.Linear(128,output_shape)

        )
        self.optimizer=optim.Adam(self.parameters(), lr=0.00002)
    def forward(self,x):
        x=self.bn1(x)
        x=self.fc1(x)
        x=self.fc1(x)
        output=self.output(x)
        return output

    def train(self):
        self.optimizer.zeros_grad()
        loss=nn.MSELoss()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')




if __name__=="__main__":
    main()
