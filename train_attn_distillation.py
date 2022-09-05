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

from teacher_forward_pass import feed_forward_teacher
from student_forward_pass import train_student

def run_trainer(train_loader, params, net_t, net_s, optimizer_s):
    import os
    from plotting import plot_loss

    dump_dir_t = 'dumps_t'
    restart_file_t = str(params.restart_file_t)

    dump_dir_s = 'dumps_s'
    restart_file_s = str(params.restart_file_s)
    
    if not os.path.exists('./'+dump_dir_t):
        os.makedirs(dump_dir_t)

    if not os.path.exists('./'+dump_dir_s):
        os.makedirs(dump_dir_s)

    encoder_t = net_t[0]
    decoder_t = net_t[1]
    conv_fb_highway_t = net_t[2]

    encoder_s = net_s[0]
    decoder_s = net_s[1]
    conv_fb_highway_s = net_s[2]

    encoder_optimizer_s = optimizer_s[0]
    decoder_optimizer_s = optimizer_s[1]
    conv_fb_highway_optimizer_s = optimizer_s[2]

    if restart_file_t!='':
        print('loading teacher encoder, decoder, conv_fb_highway')
        encoder_t.load_state_dict(torch.load(dump_dir_t+'/encoder_epoch_t_'+restart_file_t+'.pth'))
        decoder_t.load_state_dict(torch.load(dump_dir_t+'/decoder_epoch_t_'+restart_file_t+'.pth'))
        conv_fb_highway_t.load_state_dict(torch.load(dump_dir_t+'/conv_fb_highway_epoch_t_'+restart_file_t+'.pth'))
    else:
        raise Exception('Error - No teacher network restart')

    if restart_file_s!='':
        print('loading student encoder, decoder, conv_fb_highway')
        encoder_s.load_state_dict(torch.load(dump_dir_s+'/encoder_epoch_s_'+restart_file_s+'.pth'))
        decoder_s.load_state_dict(torch.load(dump_dir_s+'/decoder_epoch_s_'+restart_file_s+'.pth'))
        conv_fb_highway_s.load_state_dict(torch.load(dump_dir_s+'/conv_fb_highway_epoch_s_'+restart_file_s+'.pth'))
    else:
        print('No student network restart. Start from scratch.') 

    num_epochs = params.num_epochs

    loss_array = []
    dist_attn_loss_array = []
    dist_hint_loss_array = []

    for epoch in range(num_epochs):
        #set these to true since we are training
        #we flip them to false in the tester
        encoder_t.train()
        decoder_t.train()
        conv_fb_highway_t.train()

        encoder_s.train()
        decoder_s.train()
        conv_fb_highway_s.train()
        
        print('epoch',epoch)
        encoder_outputs_t, attn_weights_t, loss_t = feed_forward_teacher(train_loader,params,encoder_t,decoder_t,conv_fb_highway_t, epoch)
        loss_s, dist_attn_loss, dist_hint_loss = train_student(train_loader, encoder_outputs_t, attn_weights_t, params, encoder_s, decoder_s, conv_fb_highway_s, encoder_optimizer_s, decoder_optimizer_s, conv_fb_highway_optimizer_s, epoch)

        loss_array.append(loss_s)
        dist_attn_loss_array.append(dist_attn_loss)
        dist_hint_loss_array.append(dist_hint_loss)
        plot_loss(loss_array, dist_attn_loss_array, dist_hint_loss_array, 'train_s', params)

        if epoch%params.save_epoch==0:
            print('saving logs for epoch ', epoch)
            torch.save(encoder_s.state_dict(), '%s/encoder_epoch_s_%d.pth' % (dump_dir_s, epoch))
            torch.save(decoder_s.state_dict(), '%s/decoder_epoch_s_%d.pth' % (dump_dir_s, epoch))
            torch.save(conv_fb_highway_s.state_dict(), '%s/conv_fb_highway_epoch_s_%d.pth' % (dump_dir_s, epoch))
            #torch.save(postnet.state_dict(), '%s/postnet_epoch_%d.pth' %(dump_dir, epoch))
            

    
    
