#!/usr/bin/env python
import sys, os, time, copy, random, argparse
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
from torch.utils.data import random_split # Not using this
# Using Scikit-learn for dataset
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def load_rollout(filename):
    randst=10
    with open(filename, 'rb') as f:
        data=pickle.loads(f.read())
        x=data['observations']
        y=data['actions']
        x,y=shuffle(x,y,random_state=randst)
        print("Observations shape {} :: Actions shape {}".format(x.shape,y.shape))
        len_dataset=len(x)
        train_size=int(0.9*len_dataset)
        test_size=len_dataset-train_size
        x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.9,random_state=randst)
        #x_train,x_test=random_split(x,[train_size,test_size])
        #y_train,y_test=random_split(y,[train_size,test_size]) # Creates a Pytorch tensor
        print("X_train shape: {} Y_train shape: {}".format(x_train.shape,y_train.shape))
        print("X_test shape: {} Y_test shape: {}".format(x_test.shape,y_test.shape))
        #print("X_train features: ",len(x_train[0]))

        assert len(x_train)==len(y_train)
        assert len(x_test)==len(y_test)
    return x_train,y_train,x_test,y_test


class simple_neuralnet(torch.nn.Module):
    def __init__(self,input_shape,output_shape):
        super(simple_neuralnet,self).__init__()
        self.bn1=nn.Sequential(
            nn.BatchNorm1d(input_shape)
        )
        self.fc1=nn.Sequential(
            nn.Linear(input_shape,128),
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
    parser=argparse.ArgumentParser()
    parser.add_argument('envname',type=str)
    args=parser.parse_args()

    data_folder='expert_data/'
    path=os.path.join(data_folder,args.envname+".pkl")
    data=load_rollout(path)

    #main()
