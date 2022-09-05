import numpy as np
import torch
import librosa
import os
from os import listdir
from os.path import isfile,join
import re
from read_audio import read_audio
from read_audio import read_embeds
from read_audio import dump_pickles
from read_audio import dump_mels
from read_audio import read_pickles
from read_audio import create_splitfile

dataset = 'ljspeech'
read_dir = '/s/pnaray11/LJSpeech-1.1'
dump_dir_wavs = './ljspeech'
source_voices = ['wavs']
target_voices = ['']


metadata = []
with open(read_dir+"/metadata.csv", "r", encoding="utf8") as metafile:
    metadata = [line.split("|")[0] for line in metafile]

metafile.close()
print('metadata', metadata)



for voice in source_voices:
    print('processing voice ', voice)

    mels =read_audio(read_dir+'/'+voice, metadata)
    #print(padded_mels)

    #dump_pickles(dump_dir_wavs,voice+'_train',train_mels)
    #dump_pickles(dump_dir_wavs, voice+'_test', test_mels)

    dump_mels(dump_dir_wavs, mels, metadata)

    #print('Reading ', voice+'_train')
    #read_pickles(dump_dir_wavs,voice+'_train')
    #print('yes yes')

    #print('Reading', voice+'_test')
    #read_pickles(dump_dir_wavs, voice+'_test')
    #print('yes yes') 


print('dump_dir', dump_dir_wavs)
create_splitfile(dump_dir_wavs, metadata)

#train_embeds = read_embeds(read_dir+'/'+'embeddings', read_dir+'/train.txt')

#train_embeds = torch.FloatTensor(train_embeds)
#test_embeds = torch.FloatTensor(test_embeds)
#print('train_embeds.shape', train_embeds.size())
#print('test_embeds.shape', test_embeds.size())
