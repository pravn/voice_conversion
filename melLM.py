from __future__ import print_function, division
import os
import torch
#from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math

USE_CUDA=True

class Highway(nn.Module):
    def __init__(self, in_size, out_size):
        super(Highway, self).__init__()
        self.H = nn.Linear(in_size, out_size)
        self.H.bias.data.zero_()
        self.T = nn.Linear(in_size, out_size)
        self.T.bias.data.fill_(-1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        H = self.relu(self.H(inputs))
        T = self.sigmoid(self.T(inputs))
        return H * T + inputs * (1.0 - T)


class Prenet(nn.Module):
    def __init__(self, in_size, out_size):
        super(Prenet, self).__init__()
        self.bottleneck = nn.Linear(in_size, out_size)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        return self.dropout(self.relu(self.bottleneck(inputs)))

class Decoder_GRUs(nn.Module):
    def __init__(self,hidden_size,num_layers=2):
        super(Decoder_GRUs, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.GRU1 = nn.GRUCell(self.hidden_size, self.hidden_size)
        self.GRU2 = nn.GRUCell(self.hidden_size, self.hidden_size)
        self.GRU3 = nn.GRUCell(self.hidden_size, self.hidden_size)
        
        self.decoder_layers = nn.ModuleList(
            [nn.GRUCell(self.hidden_size, self.hidden_size) for _ in range(self.num_layers)])

    def forward(self, x, h1, h2, h3):
        #for i in range(self.num_layers):
        #x = self.decoder_layers[i](x) + x

        h1 = self.GRU1(x,h1) + x
        h2 = self.GRU2(h1, h2) + h1
        h3 = self.GRU3(h2, h3) + h2

        return h1, h2, h3
            

class Conv_FB_Highway(nn.Module):
    def __init__(self,input_size,seq_len,r):
        super(Conv_FB_Highway,self).__init__()
        self.input_size = input_size
        #self.seq_len = seq_len

        self.prenet = Prenet(input_size, input_size)
        
        self.conv_one = nn.Conv1d(input_size,4*input_size,kernel_size=25,padding=12)
        self.conv_three = nn.Conv1d(input_size,4*input_size,kernel_size=25,padding=12)
        self.conv_five = nn.Conv1d(input_size,4*input_size,kernel_size=25,padding=12)
    
        self.pool5 = nn.MaxPool1d(kernel_size=5,stride=1,padding=2)

        self.conv_1_ = nn.Conv1d(12*input_size,input_size,1,padding=0)
        self.conv_3_ = nn.Conv1d(12*input_size,input_size,3,padding=1)
        self.conv_5_ = nn.Conv1d(12*input_size,input_size,5,padding=2)

        self.pre_highway = nn.Linear(seq_len,seq_len)

        self.highways = nn.ModuleList(
            [Highway(input_size,input_size) for _ in range(4)])

        self.bn = nn.BatchNorm1d(4*input_size)
        self.relu = nn.ReLU()

    def forward(self,x):
        #(T,B,dim)
        x = x.transpose(0,1) #(B,T,dim)
        x = self.prenet(x) # (B,T,dim)
        #transpose to convolve in time axis
        x = x.transpose(1,2) #(B,dim,T)
        
        x1 = self.relu(self.conv_one(x))
        x1 = self.bn(x1)

        x2 = self.relu(self.conv_three(x))
        x2 = self.bn(x2)

        x3 = self.relu(self.conv_five(x))
        x3 = self.bn(x3)

        x_cat = torch.cat((x1,x2,x3),1)

        #projection from 12xinput_size -> input_size
        x_cat = self.relu(self.conv_3_(x_cat))

        x_cat = x_cat.transpose(1,2)
        x_cat = x_cat.transpose(0,1)

        #print('x_cat.size()', x_cat.size())
        
        return x_cat


#https://discuss.pytorch.org/t/how-to-reverse-a-torch-tensor/382
def reverse(x):
    #reverse first dimension
    #first - create a tensor with reversed dims 
    idx = [i for i in range(x.size(0)-1, -1, -1)]
    idx = torch.cuda.LongTensor(idx)

    #create tensor with reversed dims
    inverted_tensor = x.index_select(0, idx)

    return inverted_tensor

class BiGRU(nn.Module):
    def __init__(self,input_size, concat_hidden_size):
        super(BiGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = concat_hidden_size//2

        #hand designed, so bidirectional is set to False
        self.GRU_f = nn.GRU(input_size, self.hidden_size, bidirectional=False, batch_first=False)
        self.GRU_b = nn.GRU(input_size, self.hidden_size, bidirectional=False, batch_first=False)

    def forward(self, x):
        # x: (B,T,2*dim) 
        batch_size = x.size(0)
        
        #make sure that first dimension is timesteps
        #flip to (T,B,2*dim)
        x_f = x.transpose(0,1)
        x_b = reverse(x_f)

        h_f = Variable(torch.cuda.FloatTensor(1, batch_size, self.hidden_size).zero_())
        h_b = torch.zeros_like(h_f)

        #y_f: (T, B, dim/2)
        #y_b: (T, B, dim/2)
        y_f, h_f = self.GRU_f(x_f, h_f)
        y_b, h_b = self.GRU_b(x_b, h_b)

        #need to reverse again before concatenating
        y_b = reverse(y_b)

        #(T,B,dim/2)->(T,B,dim)
        y = torch.cat((y_f, y_b), 2)

        return y.transpose(0,1) # (B,T,dim)
        
    
class Pyramidal_GRU(nn.Module):
    def __init__(self,input_size,concat_hidden_size,stack_size):
        super(Pyramidal_GRU, self).__init__()

        print('Initializing Pyramid')
        self.input_size = input_size
        self.hidden_size = concat_hidden_size//2
        self.stack_size = stack_size

        self.gru0 = nn.GRU(input_size, self.hidden_size, batch_first=True,bidirectional=True)
        self.pyramid = nn.ModuleList(
            [BiGRU(4 * self.hidden_size, 2 * self.hidden_size) for _ in range(stack_size)])
        
        
    def forward(self, input):
        #(B,T,input_dim)-> (B,T,dim)
        x, hidden = self.gru0(input)
        seq_len = x.size(1)

        for i in range(self.stack_size):
            x = x.contiguous().view(-1, int(seq_len/2), int(4*self.hidden_size))
            seq_len /= 2
            x = self.pyramid[i](x)

        return x

class Pyramidal_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, stack_size=2):
        super(Pyramidal_Encoder, self).__init__()
        print("Initializing Pyramidal Encoder ")
        
        self.bidirectional = False
        self.input_size = input_size 
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.stack_size = stack_size
        
        self.pyramid = Pyramidal_GRU(input_size, hidden_size, stack_size)

    def forward(self, input):
        #(T,B,dim) -> (B,T, dim)
        x = input.transpose(0,1) #pyramid has batch_first flag on
        x = self.pyramid(x)
        return x.transpose(0,1) #flip back to (seq_len, batch_size, hidden_size)

        
class Encoder(nn.Module):
    def __init__(self,input_size,hidden_size,batch_size,num_layers=1):
        super(Encoder,self).__init__()
        print('Initializing plain Encoder with packed sequence api')

        self.bidirectional=True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers  = num_layers
        self.lstm = nn.LSTM(self.input_size,self.hidden_size, self.num_layers,bidirectional=self.bidirectional)
        #self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, bidirectional=self.bidirectional)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()


        if(self.bidirectional==True):
            self.layers = 2
        else:
            self.layers = 1

        self.fc = nn.Linear(self.layers * self.hidden_size, self.hidden_size)

        self.initHidden()

    def forward(self,x, seq_len):
        h = Variable(torch.zeros(self.layers*self.num_layers,self.batch_size,self.hidden_size)).cuda()
        c = Variable(torch.zeros(self.layers*self.num_layers,self.batch_size,self.hidden_size)).cuda()

        #seq_len = torch.stack(seq_len)
        #packed = torch.nn.utils.rnn.pack_padded_sequence(x,seq_len)


        #encoder RNN
        output, hidden = self.lstm(x,(h,c))

        #unpack
        #output,output_lengths = torch.nn.utils.rnn.pad_packed_sequence(output)
        return self.relu(self.fc(output)) 
        #return output, hidden, self.relu(self.fc(hidden[0]))
        
    def initHidden(self):
        return torch.zeros(self.layers,self.batch_size,self.hidden_size)


class EncoderCell(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(EncoderCell,self).__init__()

        print('Initializing EncoderCell')
        self.input_size = input_size
        self.hidden_size = hidden_size 
        self.lstmcell = nn.LSTMCell(self.input_size,self.hidden_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self,x,hidden,c):
        hidden,c = self.lstmcell(x,(hidden,c))
        output = self.tanh(self.fc(hidden))
        return output,hidden,c



            

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        
        self.method = method
        self.hidden_size = hidden_size
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        
        if self.method == 'general':
            self.processed_memory = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.processed_query = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.attn =  nn.Linear(self.hidden_size, 1, bias=False)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size, 1, bias=False)
            self.processed_memory = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.processed_query = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.v = nn.Parameter(torch.cuda.FloatTensor(self.hidden_size))
            self.content_layer = nn.Linear(2*self.hidden_size, self.hidden_size)
            #torch.nn.init.xavier_normal(self.v)

    def forward(self, hidden, encoder_outputs, processed_location, mask):
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)

        # Create variable to store attention energies
        #attn_energies = Variable(torch.zeros(this_batch_size, max_len)) # B x S
        attn_energies = Variable(torch.zeros(max_len, this_batch_size)) # S x B

        if USE_CUDA:
            attn_energies = attn_energies.cuda()

        #encoder : T x B x S 
        h = hidden  #B X S

        encoder_outputs = encoder_outputs.transpose(0,1)

        attn_energies = self.score(encoder_outputs, hidden, processed_location)
        #attn_energies += mask
        mask = mask.bool()

        
        attn_energies = attn_energies.masked_fill_(mask, -float("inf"))
        return F.softmax(attn_energies,dim=1).unsqueeze(1), attn_energies
    
    def score(self, encoder_output, hidden, processed_location):
        if self.method == 'dot':
            batch_size = encoder_output.size(0)
            hidden_size = self.hidden_size
            h = hidden
            h = h.view(batch_size,h.size(1), 1)

            energy = torch.bmm(encoder_output,h)

            return energy.squeeze(2)


        #Luong's 'general' multiplicative attention 
        if self.method == 'general':
            #encoder outputs (B,T,dim)
            processed_memory = encoder_output
            #decoder hidden 'query' vector (B,dim)
            processed_query = self.relu(self.processed_query(hidden)) #(B,dim)

            processed_query = processed_query.unsqueeze(2) #(B,dim,1)

            #Batch dot product: (B,T,dim) . (B, dim, 1) -> (B,T,1)
            content = torch.bmm(processed_memory, processed_query) #(B,T,1)
            

            #(B,T,dim) -> (B,T,1)
            energy = self.attn(self.tanh(content + processed_location))
            return energy.squeeze(2)

        elif self.method == 'concat':
            #hidden: (B,dim)
            #encoder_output: (B,T,dim)

            processed_memory = self.processed_memory(encoder_output) #(B,T,dim)
            processed_query = self.processed_query(hidden) #(B,dim)
            processed_query = processed_query.unsqueeze(1) # (B,1,dim) - now broadcastable

            # (B,1,dim)-> B(T,dim)
            processed_query = processed_query + processed_memory.data.zero_()
            
            # (B,T,dim)->(B,T,2*dim)
            #content = torch.cat((processed_query,processed_memory),2)

            #(B,T,2*dim)->(B,T,dim)
            #content = self.content_layer(content)

            #(B,T,dim)->(B,T,1)
            energy = self.attn(self.tanh(processed_memory + processed_query + processed_location))

            #(B,T,dim)->(B,dim,T)
            energy = energy.transpose(1,2)

            # (dim)->(B,1,dim)
            #v = self.v.expand(energy.size(0),1,self.v.size(0))

            #output: (B,1,T)
            #energy = torch.bmm(v,energy)
            return energy.squeeze(1)


class NoAttnDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1,max_length=500):
        super(NoAttnDecoder, self).__init__()

        # Define parameters
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length
        
        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        #self.gru = nn.GRU(hidden_size+output_size, hidden_size, n_layers, dropout=dropout_p)
        self.grucell = nn.GRUCell(hidden_size+output_size,hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.lstmcell = nn.LSTMCell(hidden_size+output_size,hidden_size)
        

    def forward(self,frame, last_hidden, encoder_outputs, decoder_c):
        frame = self.dropout(frame)

        context = encoder_outputs[-1]

        rnn_input = torch.cat((frame,context),1)
        hidden, decoder_c = self.lstmcell(rnn_input,(last_hidden,decoder_c))

        output = self.out(hidden)
        output = self.relu(output)

        return output, hidden, decoder_c


class LocationSensitiveAttention(nn.Module):
    def __init__(self,dim):
        super(LocationSensitiveAttention, self).__init__()

        self.num_filters = 20
        self.kernel_size = 7
        self.padding = (self.kernel_size-1)//2
        #self.location_conv = nn.Conv1d(1, self.num_filters, self.kernel_size, 1, self.padding) # same padding
        self.location_conv = nn.Conv1d(1, self.num_filters, self.kernel_size, 1, self.padding,bias=False) # same padding 
        self.location_layer = nn.Linear(self.num_filters, dim, bias=False)
        #self.location_layer = nn.Linear(dim, dim,bias=None)

    def forward(self, attn):
        #attn -> (B,T)
        attn = attn.unsqueeze(1) # (B,1,T)
        attn = self.location_conv(attn) # (B,num_filters,T)
        attn = attn.transpose(1,2) # (B,T,num_filters)
        attn = self.location_layer(attn) #(B,T,dim)
        return attn
        
        
        
class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, r=2, n_layers=1, dropout_p=0.1,max_length=500):
        super(BahdanauAttnDecoderRNN, self).__init__()
        
        # Define parameters
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.r = r
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length
        
        # Define layers
        self.prenet = nn.Sequential(Prenet(r*output_size, 256),
                                    Prenet(256, 128))

        #Decoder GRUs
        self.decoder_grus = Decoder_GRUs(hidden_size)
        
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.location_attn = LocationSensitiveAttention(hidden_size)
        self.attn = Attn('general', hidden_size)
        self.attn_combine = nn.Linear(hidden_size+128,hidden_size+128)
        #self.gru = nn.GRU(hidden_size+output_size, hidden_size, n_layers, dropout=dropout_p)
        self.project_embedding = nn.Linear(256,128)
        self.grucell = nn.GRUCell(hidden_size+128,hidden_size)
        self.out = nn.Linear(hidden_size, r*output_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        #self.GRU1 = nn.GRUCell(hidden_size, hidden_size)
        #self.GRU2 = nn.GRUCell(hidden_size, hidden_size)
    
    def forward(self, frame, last_hidden, encoder_outputs, prev_attn, h1, h2, h3, mask, embedding):

        #T: encoder top layer timesteps
        #encoder_outputs: (T, B, dim)
        #last_hidden: (B, dim)
        #frame: (B, input_size)
        #prev_attn: (B, T)
        #mask: (B, T)

        #Pieces
        #1. Pre-net 
        #2. Attention RNN (the Bahdanau piece + location sensitive attention)
        #3. Decoder RNN layers
        #4. Project to mel 
        
        #(B,input_size)->(B,prenet_size)
        frame = self.prenet(frame)


        #prev_attn: (B, T)
        #processed_location: (B, T, dim)
        processed_location = self.location_attn(prev_attn)

        # Calculate attention weights and apply to encoder outputs
        attn_weights, attn_energies = self.attn(last_hidden, encoder_outputs, processed_location, mask)

        #print('encoder_outputs.size()', encoder_outputs.size())
        encoder_outputs = encoder_outputs.transpose(0,1)
        context = torch.bmm(attn_weights, encoder_outputs)
        context = context.transpose(0, 1) # 1 x B x N

        embedding = self.relu(self.project_embedding(embedding))

        frame = frame.unsqueeze(0)
        embedding = embedding.unsqueeze(0)
        
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat((frame+embedding,context), 2)
        rnn_input = self.attn_combine(rnn_input)
        rnn_input = self.relu(rnn_input)

        #This is the attention RNN in the Tacotron paper
        hidden = self.grucell(rnn_input.squeeze(0), last_hidden.squeeze(0))

        #Decoder GRU layers (2 in total in the Tacotron paper)
        h1, h2, h3 = self.decoder_grus(hidden, h1, h2, h3)


        output = self.out(h3)
        output = self.relu(output) #(B,r*mel_dim) 

        output = output.view(self.r,-1,self.output_size) 
        
        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights, attn_energies, h1, h2, h3
    

def get_conv_fb_highway(params,seq_len):
    cbh = Conv_FB_Highway(params.input_size,seq_len,params.r)
    return cbh

def get_encoder(params):
    #print('vanilla lstm')
    #e = Encoder(params.input_size, params.hidden_size, params.batch_size, params.num_layers)
    
    print('pyramidal GRU')
    e = Pyramidal_Encoder(params.input_size, params.hidden_size, params.batch_size, params.stack_size)
    
    return e

def get_decoder(params):
    print('Bahdanau Attention Decoder')
    #print("No Attention Decoder")
    d = BahdanauAttnDecoderRNN(params.hidden_size, params.input_size, params.r)
    #d = NoAttnDecoder(params.hidden_size,params.input_size)
    return d

