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

def test(test_loader,params,encoder,decoder,conv_fb_highway, epoch):
    
    from train_attn import mse_loss
    from plotting import plot_mel
    from plotting import plot_attn
    from plotting import plot_loss
    
    test_loss = 0
    j = 0

    for i, [sample,embedding] in enumerate(test_loader):
        #print('l', len(train_loader))
        j+=1
        if(j==len(test_loader)):
            break

       
        #print('batch ', i)
        src = sample['source']
        tgt = sample['target']
        mask = sample['mask']

        embedding = embedding['target']
        embedding = embedding.cuda()

        #print('src.size()', src.size())
        #print('mask.size()', mask.size())

        #src_seqlen = seq_len['source']
        #tgt_seqlen = seq_len['target']

        #seq_quads = sorted(zip(src,src_seqlen,tgt,tgt_seqlen),key=lambda p:p[1],reverse=True)

        #src,src_seqlen, tgt,tgt_seqlen = zip(*seq_quads)

        #src = torch.stack(src)
        #src_seq_len = torch.stack(src_seqlen)
        #tgt = torch.stack(tgt)
        #tgt_seqlen = torch.stack(tgt_seqlen)

        #src: (B,dim,T)
        src = src.transpose(0,2)
        src = src.transpose(1,2)
        #src: (T,B,dim)

        #print('src.size()', src.size())

        src = Variable(src.float()).cuda()

        #mask: (B, T)
        mask = Variable(mask.float()).cuda()
        #print('mask.size()', mask.size())

        src0 = src

        print('src.size()', src.size())
        src = conv_fb_highway(src)

        tgt = tgt.transpose(0,2)
        tgt = tgt.transpose(1,2)
        tgt = Variable(tgt.float()).cuda()

        #max_src_length = max(src_seqlen)
        
        #HACK
        max_src_length = src.size(0)//2**params.stack_size #pyramid 

        enc_h = torch.zeros(params.batch_size,params.hidden_size).cuda()
        enc_h = Variable(enc_h)

        enc_c = torch.zeros(params.batch_size,params.hidden_size).cuda()
        enc_c = Variable(enc_c)
        
        encoder_outputs = torch.zeros((max_src_length,params.batch_size,params.hidden_size)).cuda()
        encoder_outputs = Variable(encoder_outputs)
        encoder_outputs = encoder(src)

        #print('encoder_outputs.size()', encoder_outputs.size())
        #print('mask.size()', mask.size())
        
        #max_target_length = max(tgt_seqlen)
        max_target_length = int(params.max_tgt_length)

        all_decoder_outputs = Variable(torch.zeros(max_target_length,params.batch_size, params.input_size)).cuda()
        decoder_attns = torch.zeros(max_target_length//params.r,params.batch_size, max_src_length)

        decoder_attn_energies = torch.zeros(max_target_length//params.r, params.batch_size, max_src_length)

        enc_out = torch.zeros(encoder_outputs.size())

        dec_hidden = torch.zeros(max_target_length//params.r,params.batch_size, params.hidden_size)
        
        GO = torch.zeros(params.batch_size,params.r*params.input_size)
        STOP = torch.ones(params.batch_size,params.input_size)

        decoder_input = Variable(GO)
        decoder_input = decoder_input.cuda()
        decoder_hidden = encoder_outputs[-1]

        h1 = encoder_outputs[-1]
        h2 = encoder_outputs[-1]

        decoder_c = torch.zeros(params.batch_size,params.hidden_size).cuda()
        decoder_c= Variable(decoder_c)
        
        teacher_forcing = False

        decoder_attn_weights = torch.zeros(params.batch_size, max_src_length)
        decoder_attn_weights = Variable(decoder_attn_weights).cuda()
        
        for t in range(max_target_length//params.r):
            decoder_output,decoder_hidden, decoder_attn_weights, attn_energies, h1, h2 = decoder(decoder_input,decoder_hidden,encoder_outputs, decoder_attn_weights, h1, h2, mask, embedding)

            for i in range(params.r):
                all_decoder_outputs[params.r*t+i] = decoder_output[i]

            
            if teacher_forcing: #never gets here in inference mode
                #decoder_input = tgt[params.r*t+i]#next input is current target
                decoder_input = tgt[t:t+params.r]
                decoder_input = decoder_input.view(-1,params.r*params.input_size)
            else: #always uses this in inference mode
                #decoder_input = decoder_output[-1]
                decoder_input = decoder_output
                decoder_input = decoder_input.view(-1,params.r*params.input_size)
                
            decoder_attn_weights = decoder_attn_weights.squeeze(1)

            decoder_attns[t] = decoder_attn_weights.data.cpu()
            decoder_attn_energies[t] = attn_energies.data.cpu()
            dec_hidden[t] = decoder_hidden


        loss = mse_loss(all_decoder_outputs,tgt) #log loss inside function

        #loss.backward()

        test_loss += loss.item()
        
        decoder_attns = decoder_attns.transpose(0,1)
        recon = all_decoder_outputs.transpose(0,1)
        recon = recon.transpose(1,2)
        target = tgt.transpose(0,1)
        target = target.transpose(1,2)
        dec_hidden = dec_hidden.data.cpu().transpose(0,1).detach()

        enc_out = encoder_outputs.data.cpu().transpose(0,1).detach()


        #plot things
        if(j==1):
            plot_attn(decoder_attns[1].numpy(), ' decoder_attns_test_'+str(epoch),params)
            plot_mel(recon[1].data.cpu().numpy(), ' recon_test_'+str(epoch),params)
            
            plot_mel(target[1].transpose(0,1)[:recon[1].size(1)].transpose(0,1).data.cpu().numpy(),
                     ' target_test_'+str(epoch),params)
            
            plot_mel(src0.transpose(0,1)[1].transpose(0,1).data.cpu().numpy(),
                     ' source_test_'+str(epoch),params)

    print('test_loss', test_loss)
    
    return test_loss

            
        
def run_tester(test_loader,params,encoder,decoder,conv_fb_highway, epoch):
    from plotting import plot_loss
    loss_array = []

    encoder.training=False
    decoder.training=False
    conv_fb_highway.training=False

    loss = test(test_loader,params,encoder,decoder,conv_fb_highway, epoch)
    loss_array.append(loss)
    plot_loss(loss_array, 'test', params)
        
    
    
