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


criterion_L1=nn.L1Loss(reduction='sum')
criterion_L2 = nn.MSELoss(reduction='sum')


def get_attn_penalty(attn_weights,
                     decoder_step, encoder_maxstep, decoder_maxstep):

    attn_weights = attn_weights.squeeze(1)
    nu = 0.2
    lambda_attn = 0.1
    encoder_steps = torch.arange(0,encoder_maxstep).float().cuda()

    #penalty_term = attn_weights

    #for each decoder timestep -> (B, encoder_maxstep)
    guided_attn_term = 1-torch.exp(-(encoder_steps/(encoder_maxstep-1) - (decoder_step-1)/(decoder_maxstep-1))
                                   *(encoder_steps/(encoder_maxstep-1) - (decoder_step-1)/(decoder_maxstep-1))
                                   /(2.0 * nu * nu))

    #print('attn_weights.size()', attn_weights.size())
    #print('guided_attn_term.size()', guided_attn_term.size())
     
    penalty_term = lambda_attn * attn_weights * guided_attn_term

    #print('penalty_term.size()', penalty_term.size())

    return penalty_term

#attn_penalty = Variable(torch.zeros(max_target_length/int(params.r), params.batch_size)).cuda()
#In decoder timestep loop
#for t in range(decoder_timesteps):
#  attn_penalty_term = get_attn_penalty(...) #-> (B,encoder_maxstep)

def L1Loss(input,target):
    t = target[:input.size(0)][:][:]
    t = t.transpose(0,1)
    inp = input.transpose(0,1)
    L = criterion_L1(inp, t)

    return L

def L2Loss(input,target):
    t = target[:input.size(0)][:][:]
    t = t.transpose(0,1)
    inp = input.transpose(0,1)
    L = criterion_L2(inp, t)

    return L


def mse_loss(input,target):
    eps = 1e-3
    #eps = Variable(torch.cuda.FloatTensor(input.size(0),input.size(1),input.size(2)).fill_(eps))
    t = target[:input.size(0)][:][:]

    #print('input.size()', input.size(), t.size())
    
    #print('absinput', torch.abs(input+eps))
    #print('abstgt', torch.log(torch.abs(t+eps)))
    inp = torch.log(input+eps)
    t   = torch.log(t+eps)
    v = torch.abs(inp-t).pow(2)
    #print('n',torch.norm(v,2))
    #L = torch.norm(torch.log(v),2)/input.size(1)
    #L = torch.sum(v).pow(0.5)
    L = torch.sum(v)
    return L

def mse_loss2(input,target):
    eps = 1e-3
    #eps = Variable(torch.cuda.FloatTensor(input.size(0),input.size(1),input.size(2)).fill_(eps))
    t = target[:input.size(0)][:][:]

    #print('input.size()', input.size(), t.size())
    
    #print('absinput', torch.abs(input+eps))
    #print('abstgt', torch.log(torch.abs(t+eps)))
    inp = input+eps
    t   = t+eps
    v = torch.abs(inp-t).pow(2)
    #print('n',torch.norm(v,2))
    #L = torch.norm(torch.log(v),2)/input.size(1)
    #L = torch.sum(v).pow(0.5)
    L = torch.sum(v)
    return L


def get_schedule(epoch,schedule_type):
    if schedule_type == 'linear':
        return max(0, 1-0.01*epoch)
    
def train(train_loader,params,encoder,decoder,conv_fb_highway, 
          encoder_optimizer,decoder_optimizer,conv_fb_highway_optimizer, schedule,epoch):

    from plotting import plot_mel
    from plotting import plot_attn
    from plotting import plot_loss

    train_loss = 0
    j = 0
    
    for i, [sample,embedding] in enumerate(train_loader):
        #print('l', len(train_loader))
        j+=1
        if(j==len(train_loader)):
            break


        print('batch', i)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        conv_fb_highway_optimizer.zero_grad()
        #postnet_optimizer.zero_grad()

        src = sample['source']
        tgt = sample['target']
        mask = sample['mask']

        #print('src.size()', src.size())
        #print('tgt.size()', tgt.size())

        embedding = embedding['target']
        embedding = embedding.cuda()

        #src: (B,dim,T)
        src = src.transpose(0,2)
        src = src.transpose(1,2)
        #src: (T,B,dim)

        src = Variable(src.float()).cuda()

        
        #mask: (B, T)
        mask = Variable(mask.float()).cuda()
        #print('mask.size()', mask.size())

        src0 = src
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

        decoder_attn_energies = torch.zeros(max_target_length, params.batch_size, max_src_length)

        enc_out = torch.zeros(encoder_outputs.size())

        dec_hidden = torch.zeros(max_target_length,params.batch_size, params.hidden_size)

        GO = torch.zeros(params.batch_size,params.r*params.input_size)
        STOP = torch.ones(params.batch_size,params.input_size)

        decoder_input = Variable(GO)
        decoder_input = decoder_input.cuda()
        decoder_hidden = encoder_outputs[-1]

        h1 = encoder_outputs[-1]
        h2 = encoder_outputs[-1]
        h3 = encoder_outputs[-1]

        #h1 = Variable(torch.zeros(params.batch_size, 1024)).cuda()
        #h2 = Variable(torch.zeros(params.batch_size, 1024)).cuda()
        

        decoder_c = torch.zeros(params.batch_size,params.hidden_size).cuda()
        decoder_c= Variable(decoder_c)
        
        if random.random() < schedule:
            teacher_forcing = True
        else:
            teacher_forcing = False

        teacher_forcing = True

        decoder_attn_weights = torch.zeros(params.batch_size, max_src_length)
        decoder_attn_weights = Variable(decoder_attn_weights).cuda()

        attn_penalty = []
        
        for t in range(max_target_length//params.r):
            decoder_output,decoder_hidden, decoder_attn_weights, attn_energies, h1, h2, h3= decoder(decoder_input,decoder_hidden,encoder_outputs, decoder_attn_weights, h1, h2, h3,  mask, embedding)

            attn_penalty += [get_attn_penalty(decoder_attn_weights, t+1,
                                              max_src_length, max_target_length)]

            for i in range(params.r):
                all_decoder_outputs[params.r*t+i] = decoder_output[i]

            if teacher_forcing:
                #decoder_input = tgt[params.r*t+i]#next input is current target
                decoder_input = tgt[t:t+params.r]
                decoder_input = decoder_input.view(-1,params.r*params.input_size)
            else:
                #decoder_input = decoder_output[-1]
                decoder_input = decoder_output
                decoder_input = decoder_input.view(-1,params.r*params.input_size)
                
            decoder_attn_weights = decoder_attn_weights.squeeze(1)

            decoder_attns[t] = decoder_attn_weights.data.cpu()
            decoder_attn_energies[t] = attn_energies.data.cpu()
            dec_hidden[t] = decoder_hidden


        attn_penalty = torch.stack(attn_penalty)
        attn_penalty = attn_penalty.mean()

        loss = L1Loss(all_decoder_outputs,tgt) #log loss inside function

        #postnet_residual = postnet(all_decoder_outputs.transpose(1,2))

        #postnet_output = postnet_residual.transpose(1,2) + all_decoder_outputs

        

        #postnet_loss = L2Loss(postnet_output, tgt)
        #postnet_loss = 0.1*L2Loss(postnet_output, tgt)
        
        

        #attn_penalty.backward(retain_graph=True)
        loss.backward()
        #postnet_loss.backward()
        

        #train_loss += postnet_loss.item()
        train_loss += loss.item()


        clipping_value = 1#arbitrary number of your choosing
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), clipping_value)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), clipping_value)
        torch.nn.utils.clip_grad_norm_(conv_fb_highway.parameters(), clipping_value)
        #torch.nn.utils.clip_grad_norm_(postnet.parameters(), clipping_value)

        encoder_optimizer.step()
        decoder_optimizer.step()
        conv_fb_highway_optimizer.step()
        #postnet_optimizer.step()

        decoder_attns = decoder_attns.transpose(0,1)
        #recon = postnet_output.transpose(0,1)
        recon = all_decoder_outputs.transpose(0,1)
        recon = recon.transpose(1,2)
        target = tgt.transpose(0,1)
        target = target.transpose(1,2)
        dec_hidden = dec_hidden.data.cpu().transpose(0,1).detach()

        enc_out = encoder_outputs.data.cpu().transpose(0,1).detach()



        #plot things
        if(j==1):
            plot_attn(decoder_attns[1].numpy(), 'decoder_attns_'+str(epoch), params)
            plot_mel(recon[1].data.cpu().numpy(), ' recon_train_'+str(epoch), params)
            plot_mel(target[1].transpose(0,1)[:recon[1].size(1)].transpose(0,1).data.cpu().numpy(),
                     ' target_train_'+str(epoch), params)
            plot_mel(src0.transpose(0,1)[1].transpose(0,1).data.cpu().numpy(),
                     ' source_train_'+str(epoch), params)


    print('train_loss', train_loss)
    
    return train_loss

            

        

def run_trainer(train_loader, test_loader, params,encoder,decoder,conv_fb_highway, encoder_optimizer,decoder_optimizer,conv_fb_highway_optimizer):
    import os
    from plotting import plot_loss
    from test_attn import run_tester

    dump_dir = 'dumps'
    restart_file = str(params.restart_file)
    #restart_file = str(59)
    
    if not os.path.exists('./'+dump_dir):
        os.makedirs(dump_dir)


    if restart_file!='':
        print('loading encoder, decoder, conv_fb_highway')
        encoder.load_state_dict(torch.load(dump_dir+'/encoder_epoch_'+restart_file+'.pth'))
        decoder.load_state_dict(torch.load(dump_dir+'/decoder_epoch_'+restart_file+'.pth'))
        conv_fb_highway.load_state_dict(torch.load(dump_dir+'/conv_fb_highway_epoch_'+restart_file+'.pth'))
        #postnet.load_state_dict(torch.load(dump_dir+'/postnet_epoch_'+restart_file+'.pth'))
    
    num_epochs = params.num_epochs

    loss_array = []

    #for param_group in encoder_optimizer.param_groups:
    #    param_group['lr'] = 1e-5

    #for param_group in decoder_optimizer.param_groups:
    #    param_group['lr'] = 1e-5

    

    for epoch in range(num_epochs):
        #set these to true since we are training
        #we flip them to false in the tester
        encoder.train()
        decoder.train()
        conv_fb_highway.train()
        #postnet.training = True
        
        print('epoch',epoch)
        schedule = get_schedule(epoch, 'linear')
        schedule = 1.0
        print('schedule', schedule)
        loss = train(train_loader,params,encoder,decoder,conv_fb_highway, encoder_optimizer,decoder_optimizer,conv_fb_highway_optimizer, schedule,epoch)

        test_loss = run_tester(test_loader, params, encoder, decoder, conv_fb_highway, epoch)

        loss_array.append(loss)
        plot_loss(loss_array, 'train', params)

        if epoch%params.save_epoch==0:
            print('saving logs for epoch ', epoch)
            torch.save(encoder.state_dict(), '%s/encoder_epoch_%d.pth' % (dump_dir, epoch))
            torch.save(decoder.state_dict(), '%s/decoder_epoch_%d.pth' % (dump_dir, epoch))
            torch.save(conv_fb_highway.state_dict(), '%s/conv_fb_highway_epoch_%d.pth' % (dump_dir, epoch))
            #torch.save(postnet.state_dict(), '%s/postnet_epoch_%d.pth' %(dump_dir, epoch))
            

    
    
