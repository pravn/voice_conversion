import numpy as np
import librosa
import os
from os import listdir
from os.path import isfile,join
import re
import pickle
import gzip
import librosa
import audio
from hparams import hparams

def dump_pickles(dir,name,mels):
    dump_dir = dir+'/mels'

    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)
    
    data = {'name':name,'mels':mels}
    gzip_file = dump_dir+'/'+name+'.pkl.gz'
    pickle.dump(data,gzip.open(gzip_file,'wb'))

def read_pickles(dir,file_name):
    dump_dir = dir+'/mels'
    gzip_file_name = dump_dir+'/'+file_name+'.pkl.gz'
    data = pickle.load(gzip.open(gzip_file_name,'r'))
    return data


def dump_mels(dir, mels, metadata):

    dump_dir = dir+'/mels'
    
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)
    
    for i in range(len(metadata)):
        filepath = dump_dir+'/'+metadata[i]+'.npy'
        np.save(filepath, mels[i])



def read_mels(path, metadata):
    import numpy as np

    mels = []

    with open(metadata, 'r') as metafile:
        entries = [entry.split('\n')[0] for entry in metafile]
    metafile.close()
    
    for file in entries:
        m = np.load(path+'/'+file+'.npy')
        mels.append(m)

    return mels

def read_postnet_mels_and_tags(path, run_type, metadata):
    import numpy as np

    mels = []

    with open(metadata, 'r') as metafile:
        entries = [entry.split('\n')[0] for entry in metafile]
    metafile.close()
    
    for file in entries:
        m = np.load(path+'/'+run_type+'/'+file+'.npy')
        mels.append(m)

    tags = entries

    return mels, tags


def read_mels_and_tags(path, metadata):
    import numpy as np

    mels = []

    with open(metadata, 'r') as metafile:
        entries = [entry.split('\n')[0] for entry in metafile]
    metafile.close()
    
    for file in entries:
        m = np.load(path+'/'+file+'.npy')
        mels.append(m)

    tags = entries

    return mels, tags



def read_mels_libritts(path, metadata):
    import numpy as np

    mels = []

    with open(metadata, 'r') as metafile:
        entries = [entry.split('\t')[0] for entry in metafile]
    metafile.close()
    
    for file in entries:
        m = np.load(path+'/'+file+'.npy')
        mels.append(m)

    return mels


        
def plot_mel(mel):
    import matplotlib.pyplot as plt
    import numpy as np
    import librosa.display 
    plt.figure(figsize=(10, 4))
    #librosa.display.specshow(librosa.power_to_db(mel,ref=np.max),
    #                 y_axis='mel', fmax=8000, x_axis='time')
    librosa.display.specshow(mel,
                     y_axis='mel', fmax=8000, x_axis='time')
    
    #plt.colorbar(format='%+2.0f dB')
    #plt.title('Mel spectrogram')
    #plt.tight_layout()
    plt.show()
    #plt.savefig('dumps2/mel_'+str(i)+'.png')
    #plt.close()

class MelSpectrogram():
    def __init__(self):

        self.method = hparams.method #only this parameter is actually needed
        self.num_mels = hparams.num_mels
        self.fmax = hparams.fmax
        #self.n_fft = hparams.fft_size
        self.hop_size = hparams.hop_size
        self.win_length = hparams.win_length
        self.sample_rate = hparams.sample_rate
        self.min_level_db = hparams.min_level_db
        self.fmin = hparams.fmin
        self.fft_size = hparams.fft_size
        self.num_mels = hparams.num_mels
        self.rescaling_max = hparams.rescaling_max
        self.rescaling = hparams.rescaling
        

    def melspectrogram(self, y, sr):
        if self.method == 'librosa':
            #m: (n_bins, seq_len)
            y = y / np.abs(y).max() 
            #m = librosa.feature.melspectrogram(y, sr, n_mels=self.n_mels, n_fft=self.n_fft,
            #                                   fmax=self.fmax,
            #                                   hop_length=self.hop_length)


            D = self._stft(y)
            S = self.amp_to_db(self._linear_to_mel(np.abs(D))) 
            #r9y9 has S = self.amp_to_db(self.linear_to_mel(np.abs(D))) - hparams.ref_level_db
            m = self.normalize(S)
            #m = S
            
        if self.method =='r9y9':
            #call r9y9's spectrogram routines
            #first normalize
            #if self.rescaling:
            y = y / np.abs(y).max() * self.rescaling_max

            #now call audio.melspectrogram

            # (D, N)

            m = audio.logmelspectrogram(y).astype(np.float32)
            m = audio._normalize(m)
            #m = audio.melspectrogram(y).astype(np.float32)

            #plot_mel(m)
            
        return m


    def preemphasis(x):
        return scipy.signal.lfilter([1, -hparams.preemphasis], [1], x)

    def stft(self,y):
        return librosa.stft(y=y,
                            n_fft=self.n_fft,
                            hop_length=self.hop_length,
                            win_length=self.win_length)

    def _stft(self,y):
        n_fft, hop_length, win_length = self._stft_parameters()
        return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    
    
    def linear_to_mel(self,spectrogram):
        return librosa.feature.melspectrogram(S=spectrogram,
                                              sr=self.sample_rate,
                                              n_fft=self.n_fft,
                                              n_mels=self.num_mels,
                                              fmin=self.fmin)

    
    def _linear_to_mel(self,spectrogram):
        _mel_basis = self._build_mel_basis()
        return np.dot(_mel_basis, spectrogram)

    def _build_mel_basis(self):
        n_fft = (hparams.num_freq - 1) * 2
        return librosa.filters.mel(hparams.sample_rate, n_fft, n_mels=hparams.num_mels)

    def _stft_parameters(self):
        n_fft = (hparams.num_freq - 1) * 2
        hop_length = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
        win_length = int(hparams.frame_length_ms / 1000 * hparams.sample_rate)
        return n_fft, hop_length, win_length

    
    def normalize(self,S):
        return np.clip((S - self.min_level_db) / -self.min_level_db, 0, 1)


    
    def amp_to_db(self,x):
        #return 20 * np.log10(np.maximum(1e-5, x))
        return 20 * np.log10(np.maximum(1e-5, x))

    

def split(metadata):

    length = len(metadata)
    train_size = int(0.9 * length)
    test_size = len(metadata) - train_size

    train = metadata[:train_size]
    test = metadata[train_size:]

    return train, test


def load_file(file, mel_wrapper):
    print('file', file)
    y, sr = librosa.load(file, hparams.sample_rate)
    m = mel_wrapper.melspectrogram(y, sr)
    tr = m.T
    tr = tr[:269]
    m = tr.T
    return m

'''
def read_audio(path, metadata):
    from hparams import hparams
    #mel hyperparameters
    mel_wrapper = MelSpectrogram()

    
    train, test = split(metadata)
    train_mels = []
    test_mels = []

    for file in train:
        m = load_file(path+'/'+file+'.wav', mel_wrapper)
        train_mels.append(m)

    for file in test:
        m = load_file(path+'/'+file+'.wav', mel_wrapper)
        test_mels.append(m)

    
    return train_mels, test_mels
'''


def read_audio(path, metadata): 
    from hparams import hparams
    mel_wrapper = MelSpectrogram()
    
    mels = []
    for i in range(len(metadata)):
        print(metadata[i])
        m = load_file(path+'/'+metadata[i]+'.wav', mel_wrapper)
        mels.append(m)

    return mels


def read_audio_arctic(path, metadata):
    from hparams import hparams
    mel_wrapper = MelSpectrogram()

    mels = []
    for i in range(len(metadata)):
        print(metadata[i])
        m = load_file(path+'/'+'arctic_'+metadata[i]+'.wav', mel_wrapper)
        mels.append(m)

    return mels
    


def read_embeds(path, metadata):
    import numpy as np

    embeds = []

    with open(metadata, 'r') as metafile:
        entries = [entry.split('\n')[0] for entry in metafile]
    metafile.close()
    
    for file in entries:
        e = np.load(path+'/'+file+'.npy')
        embeds.append(e)

    return embeds

def read_embeds_and_tags(path, metadata):
    import numpy as np

    embeds = []

    with open(metadata, 'r') as metafile:
        entries = [entry.split('\n')[0] for entry in metafile]
    metafile.close()
    
    for file in entries:
        e = np.load(path+'/'+file+'.npy')
        embeds.append(e)

    tags = entries

    return embeds, tags

def read_embeds_libritts(path, metadata):
    import numpy as np

    embeds = []

    with open(metadata, 'r') as metafile:
        entries = [entry.split('\t')[0] for entry in metafile]
    metafile.close()

    for file in entries:
        e = np.load(path+'/'+file+'.npy')
        embeds.append(e)

    return embeds

def create_splitfile(dump_dir, metadata):
    train, test = split(metadata)

    with open(dump_dir+'/train.txt', 'w') as trainfile:
        [trainfile.write(entry+"\n") for entry in train]


    with open(dump_dir+'/test.txt', 'w') as testfile:
        [testfile.write(entry+"\n") for entry in test]

    trainfile.close()
    testfile.close()
