import argparse
from transformers import BertTokenizer,AutoModel
import pickle
import tools

import load
import model
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
def getrowdata(file_path):
    data_list,tag_list = [],[]
    with open(file_path,'r') as f:
        for line in f:
            line = line.strip()
            data_list.append(line.split('\t')[0])
            tag_list.append(line.split('\t')[1])
    return data_list,tag_list        

with open("data_set.pkl",'rb') as f:
    train_set,test_set,dictf = pickle.load(f)
tag_z = train_set[2]

def force_padding(inputs,force_length,padding_word=0):
    padded_outputs = []
    for input in inputs:
        num_padding = force_length -len(input)
        if num_padding <0:
            padded_output = input[:force_length]
        else:
            padded_output = input + [int(padding_word)]*num_padding
        padded_outputs.append(padded_output)        
    return padded_outputs

padded_target = force_padding(tag_z,56)

file_path = "data/original_data/trnTweet"
data_list , tag_list = getrowdata(file_path)
padded_target = padded_target + [[0]*56] *(len(data_list)-len(padded_target))
device = torch.device('cuda:0')
sample_inputs = data_list[:100]
sample_target = padded_target[:100]
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
token_example = tokenizer(sample_inputs,padding="max_length", max_length=56,truncation =True,return_tensors='pt')
transformer_model = AutoModel.from_pretrained('bert-base-uncased')

transformer_model.train()
#transformer_outputs = transformer_model(**(token_example))
#print(transformer_outputs.keys())
max_len = max(list(len(i.split()) for i in sample_inputs))
#tokenized_data = tokenizer(data_list,padding="max_length",max_length=max_len,truncation =True,return_tensors='pt')
model = model.ModelwithTransformer(transformer_model,tokenizer,5,56,device).to(device)

batch_size = 16
def data_batch(input,target,batch_size):
    for id_start in range(0,len(input),batch_size):
        id_end = id_start +batch_size
        if id_end >len(input):
            break
        excerpt = slice(id_start,id_end)
        yield input[excerpt],target[excerpt]
#loss_fn = torch.nn.functional.cross_entropy()
lr = 1e-3
optimizer = torch.optim.Adam(model.parameters(),lr)
writer = SummaryWriter()
save_path = './checkpoints_bert'
os.makedirs(save_path,exist_ok = True)
for e in tqdm(range(10)):
    model.train()
    loss = 0.
    gz,pz = [],[]
    #for step,batch in enumerate(tqdm(tools.minibatches(train_lex,list(traind),batch_size=batch_size))):
    for step,batch in enumerate(tqdm(data_batch(data_list,padded_target,batch_size=16))):
        input_x,label_z = batch
        gz.extend(label_z)
        z = model(input_x)
        sz = z.detach().argmax(dim=-1).tolist()
        pz.extend(sz)
        z = z.permute(0,2,1)
        label_z = torch.tensor(label_z).to(device)
        loss = torch.nn.functional.cross_entropy(z,label_z)
        writer.add_scalar("training_bert/loss",loss,e*10+step)
        optimizer.zero_grad()             
        loss.backward()                
        optimizer.step()  
    des_savepath = f'{save_path}/epoch_{e:03d}.pth'   
    train_start = tools.conlleval(gz,pz)
    for key in train_start.keys():
        writer.add_scalar(f"train_bert/{key}",train_start[key],e+1)

    torch.save(model.state_dict(),des_savepath)    



