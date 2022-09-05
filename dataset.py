from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


import numpy as np
import librosa
import os
from os import listdir
from os.path import isfile,join
import re
import pickle

class Mel:
    def __init__(self,mel):
        self.mel = mel
        self.num_bins = mel.shape[0]
        self.seq_len  = mel.shape[1]


def zero_pad_end(mels):
    num_samples = len(mels)

    maxlen = 0
    for i in range(len(mels)):
        if(maxlen<mels[i].shape[1]):
            maxlen = mels[i].shape[1]

    #pad to maxlen
    for i in range(len(mels)):
        mels[i,:,:maxlen] = 0
    
        
    

class MelDataset(Dataset):
    def __init__(self,source_mels, target_mels, embedding, stack_size, maxlen_source = 0):
        self.source_mels = source_mels
        self.target_mels = target_mels
        self.stack_size = stack_size
        self.maxlen_source = maxlen_source
        self.embedding = embedding
        assert(len(source_mels)==len(target_mels))
        self.source_seq_len = []
        self.target_seq_len = []
        self.mask = []
        #self.sort_sequence()
        self.pad_sequence()

    def __len__(self):
        return len(self.source_mels)

    def __getitem__(self,idx):
        sample = {'source':self.source_mels[idx], 'target': self.target_mels[idx], 'mask': self.mask[idx]}
        #seq_len = {'source': self.source_seq_len[idx], 'target': self.target_seq_len[idx]}
        embedding = {'target': self.embedding[idx]}


        return sample, embedding

    def sort_sequence(self):
        u = sorted(zip(self.source_mels,self.target_mels), lambda p: p[0].shape[1],reverse=True)
        self.source_mels, self.target_mels = zip(*u)
            
    def pad_sequence(self):
        maxlen_source = 0
        maxlen_target = 0
        ninf = -50.0

        print('shape', self.source_mels[0].shape)
        
        num_bins = self.source_mels[0].shape[0]
        for i in range(len(self.source_mels)):
            if(maxlen_source<self.source_mels[i].shape[1]):
                maxlen_source = self.source_mels[i].shape[1]
                print('maxlen_source', maxlen_source)

        maxlen_source = max(maxlen_source, self.maxlen_source)
        #self.maxlen_source = maxlen_source
        self.maxlen_source = 501
        maxlen_source = self.maxlen_source


        print('maxlen src', maxlen_source)
            
        zeros = np.zeros((num_bins,1+maxlen_source-self.source_mels[i].shape[1]))
        stack_reduction = 2**self.stack_size
        r = stack_reduction - maxlen_source % stack_reduction
            #print('r=', r)

            
        for i in range(len(self.source_mels)):
            s = stack_reduction - self.source_mels[i].shape[1] % stack_reduction
            #STOP = 1e-4*np.ones((num_bins,stack_reduction+maxlen_source+r-self.source_mels[i].shape[1]))
            STOP = 1e-4*np.ones((num_bins, stack_reduction))

            #print('mls', maxlen_source)
            
            ZEROS = np.zeros((num_bins, maxlen_source+r-self.source_mels[i].shape[1]))
            STOP = np.concatenate((STOP,ZEROS),1)
                                   

            ZEROS = np.zeros((s+self.source_mels[i].shape[1])//stack_reduction+1)
            ONES = np.ones((r+maxlen_source-self.source_mels[i].shape[1]-s)//stack_reduction)
            
            self.mask.append(np.concatenate((ZEROS,ONES),0))
            #print('stop.shape', STOP.shape)
            self.source_mels[i] = np.concatenate((self.source_mels[i],STOP),1)
            #print('Aself.source_mels[i].shape]', self.source_mels[i].shape)


            #print('zeros', ZEROS.shape)
            #print('NINF', NINF.shape)
            #print('mask', np.concatenate((ZEROS, NINF),0).shape)
            #print('maxl', maxlen_source)




        
        
        for i in range(len(self.target_mels)):
            if(maxlen_target<self.target_mels[i].shape[1]):
                maxlen_target = self.target_mels[i].shape[1]

        r = stack_reduction - maxlen_target % stack_reduction

        for i in range(len(self.target_mels)):
            STOP = 1e-4*np.ones((num_bins, stack_reduction))
            ZEROS = np.zeros((num_bins, maxlen_target+r-self.target_mels[i].shape[1]))
            STOP = np.concatenate((STOP,ZEROS), 1)
            #STOP = 1e-4*np.ones((num_bins,stack_reduction+1+maxlen_target-self.target_mels[i].shape[1]))
            #STOP = np.ones((num_bins,1))
            self.target_mels[i] = np.concatenate((self.target_mels[i],STOP),1)
            #self.target_mels[i] = np.concatenate((self.target_mels[i],zeros),1)


class TaggedMelDataset(Dataset):
    def __init__(self,source_mels, target_mels, embedding, tags, stack_size, maxlen_source = 0):
        self.source_mels = source_mels
        self.target_mels = target_mels
        self.tags = tags
        self.stack_size = stack_size
        self.maxlen_source = maxlen_source
        self.embedding = embedding
        assert(len(source_mels)==len(target_mels))
        self.source_seq_len = []
        self.target_seq_len = []
        self.mask = []
        #self.sort_sequence()
        self.pad_sequence()

    def __len__(self):
        return len(self.source_mels)

    def __getitem__(self,idx):
        sample = {'source':self.source_mels[idx], 'target': self.target_mels[idx], 'tags': self.tags[idx], 'mask': self.mask[idx]}
        #seq_len = {'source': self.source_seq_len[idx], 'target': self.target_seq_len[idx]}
        embedding = {'target': self.embedding[idx]}


        return sample, embedding

    def sort_sequence(self):
        u = sorted(zip(self.source_mels,self.target_mels), lambda p: p[0].shape[1],reverse=True)
        self.source_mels, self.target_mels = zip(*u)
            
    def pad_sequence(self):
        maxlen_source = 0
        maxlen_target = 0
        ninf = -50.0

        print('shape', self.source_mels[0].shape)
        
        num_bins = self.source_mels[0].shape[0]
        for i in range(len(self.source_mels)):
            if(maxlen_source<self.source_mels[i].shape[1]):
                maxlen_source = self.source_mels[i].shape[1]
                print('maxlen_source', maxlen_source)

        maxlen_source = max(maxlen_source, self.maxlen_source)
        #self.maxlen_source = maxlen_source
        self.maxlen_source = 501
        maxlen_source = self.maxlen_source


        print('maxlen src', maxlen_source)
            
        zeros = np.zeros((num_bins,1+maxlen_source-self.source_mels[i].shape[1]))
        stack_reduction = 2**self.stack_size
        r = stack_reduction - maxlen_source % stack_reduction
            #print('r=', r)

            
        for i in range(len(self.source_mels)):
            s = stack_reduction - self.source_mels[i].shape[1] % stack_reduction
            #STOP = 1e-4*np.ones((num_bins,stack_reduction+maxlen_source+r-self.source_mels[i].shape[1]))
            STOP = 1e-4*np.ones((num_bins, stack_reduction))

            #print('mls', maxlen_source)
            
            ZEROS = np.zeros((num_bins, maxlen_source+r-self.source_mels[i].shape[1]))
            STOP = np.concatenate((STOP,ZEROS),1)
                                   

            ZEROS = np.zeros((s+self.source_mels[i].shape[1])//stack_reduction+1)
            ONES = np.ones((r+maxlen_source-self.source_mels[i].shape[1]-s)//stack_reduction)
            
            self.mask.append(np.concatenate((ZEROS,ONES),0))
            #print('stop.shape', STOP.shape)
            self.source_mels[i] = np.concatenate((self.source_mels[i],STOP),1)
            #print('Aself.source_mels[i].shape]', self.source_mels[i].shape)


            #print('zeros', ZEROS.shape)
            #print('NINF', NINF.shape)
            #print('mask', np.concatenate((ZEROS, NINF),0).shape)
            #print('maxl', maxlen_source)




        
        
        for i in range(len(self.target_mels)):
            if(maxlen_target<self.target_mels[i].shape[1]):
                maxlen_target = self.target_mels[i].shape[1]

        r = stack_reduction - maxlen_target % stack_reduction

        for i in range(len(self.target_mels)):
            STOP = 1e-4*np.ones((num_bins, stack_reduction))
            ZEROS = np.zeros((num_bins, maxlen_target+r-self.target_mels[i].shape[1]))
            STOP = np.concatenate((STOP,ZEROS), 1)
            #STOP = 1e-4*np.ones((num_bins,stack_reduction+1+maxlen_target-self.target_mels[i].shape[1]))
            #STOP = np.ones((num_bins,1))
            self.target_mels[i] = np.concatenate((self.target_mels[i],STOP),1)
            #self.target_mels[i] = np.concatenate((self.target_mels[i],zeros),1)

        

def make_grouping(mels):
    Mels = []

    for i in range(len(mels)):
        Mels.append(Mel(mels[i]))
    
    return Mels
