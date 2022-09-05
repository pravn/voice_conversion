from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable
import torch.nn as nn


import numpy as np
import librosa
import os
from os import listdir
from os.path import isfile,join
import re
import pickle

import random

def plot_mel(mel,tag,params):
    import matplotlib.pyplot as plt
    import numpy as np
    import librosa.display
    import os
    plt.figure(figsize=(10, 4))
    #librosa.display.specshow(librosa.power_to_db(mel,ref=np.max),
    #                         y_axis='mel', fmax=8000, x_axis='time')
    librosa.display.specshow(mel,
                             y_axis='mel', fmax=8000, x_axis='time',cmap='magma')
    #plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram '+tag)
    plt.tight_layout()

    if params.is_notebook:
        plt.show()

    if not params.is_notebook:
        if not os.path.exists(params.plots_dir):
            os.makedirs(params.plots_dir)

        plt.savefig(params.plots_dir+'/mel_'+tag+'.png')
    
    plt.close()


def plot_attn(attn,tag, params):
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    #print('plot_attn')
    fig = plt.figure(figsize=(10,4))
    #fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attn.T, origin= 'lower')
    fig.colorbar(cax)
    plt.xlabel('Decoder Timesteps')
    plt.ylabel('Encoder Timesteps')


    if params.is_notebook:
        plt.show()
    
    if not params.is_notebook:
        if not os.path.exists(params.plots_dir):
            os.makedirs(params.plots_dir)

        plt.savefig(params.plots_dir+'/attn_'+tag+'.png')

    plt.close()
    

'''    
def plot_loss(loss, dist_loss, tag, params):
    import matplotlib.pyplot as plt
    import numpy as np
    import librosa.display
    import os

    plt.figure(figsize=(10,4))
    plt.plot(loss, 'k')
    plt.plot(dist_loss, 'b')

    if params.is_notebook:
        plt.show()

    if not params.is_notebook:
        plt.savefig('./'+tag+'_loss.png')
        
    plt.close()'''

def plot_loss(loss, dist_attn_loss, dist_hint_loss, tag, params):
    import matplotlib.pyplot as plt
    import numpy as np
    import librosa.display
    import os
    
    plt.figure(figsize=(10,4))
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('Reconstruction Loss',color=color)
    ax1.plot(loss, color=color)
    #plt.cla()

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Distillation Loss')
    ax2.plot(dist_attn_loss, color=color)
    ax2.plot(dist_hint_loss, color='k')

    fig.tight_layout()
    
    if params.is_notebook:
        plt.show()

    if not params.is_notebook:
        plt.savefig('./'+tag+'_loss.png')

    plt.cla()
    plt.close()

    
def plot_loss_cycle(train_loss, cycle_loss, tag, params):
    import matplotlib.pyplot as plt
    import numpy as np
    import librosa.display
    import os
    plt.figure(figsize=(10,4))
    plt.plot(train_loss, 'k')
    plt.plot(cycle_loss, 'b')
    #plt.plot(cycle_loss, 'b')

    if params.is_notebook:
        plt.show()

    if not params.is_notebook:
        plt.savefig('./'+tag+'_loss.png')
        
    plt.close()
