from torch import nn
import torch
import load
import tools


class Model(nn.Module):
    def __init__(self,num_hiddens,ny,nz,emb,**kwargs):
        super(Model,self).__init__(**kwargs)
        self.embw = nn.Parameter(emb)
        #self.register_parameter("wemb",nn.Parameter(self.embw))
        self.num_hiddens = num_hiddens
        self.layer1 = torch.nn.RNN(900,num_hiddens,batch_first=True)
        self.layer2 = torch.nn.RNN(num_hiddens,num_hiddens,batch_first = True)
        #self.layer3 = torch.nn.GRU(num_hiddens,num_hiddens)
        self.linear1 = torch.nn.Linear(num_hiddens,ny)
        self.linear2 = torch.nn.Linear(num_hiddens,nz)
        #self.linear3 = torch.nn.Linear(num_hiddens,nz)
        self.dropout = nn.Dropout(0.5)

    def forward(self,cwards):
        device = self.embw.device
        batch_size = len(cwards)
        cwards = load.pad_sentences(cwards)
        cwards=torch.tensor(tools.contextwin_2(cwards,3)).to(device)
        cwards = nn.functional.embedding(cwards,self.embw)
        cwards = cwards.flatten(2)
        cwards = self.dropout(cwards)
        state1 = self.begin_state(batch_size).to(device)
        state2 = self.begin_state(batch_size).to(device)    
        y,state1 = self.layer1(cwards,state1)
        z,state2 = self.layer2(y,state2)
        #z,state3 = self.layer3(_,state3)
        #y = y.reshape((-1,self.num_hiddens))
        #z = z.reshape((-1,self.num_hiddens))  
        y = self.linear1(y)
        z = self.linear2(z)
       
        self.y_pred = torch.argmax(y,-1).reshape((batch_size,-1))
        self.z_pred = torch.argmax(z,-1).reshape((batch_size,-1)) 
        return y,state1,z,state2

    def begin_state(self,batch_size):
        return torch.zeros((1,batch_size,self.num_hiddens))
    def begin_state2(self,batch_size,num_hiddens):
        return torch.zeros((2,batch_size,num_hiddens))    
    def sz_pred(self):
        return  self.z_pred  



class ModelwithTransformer(nn.Module):
    def __init__(self,transformer_model,tokenizer,nz,max_len,device) -> None:
        super().__init__()
        self.transformer = transformer_model
        self.tokenizer = tokenizer
        self.max_length = max_len
        self.linear = nn.Linear(768,nz)
        self.device = device

    def forward(self,x):
        token_x = self.tokenizer(x,padding="max_length", max_length=self.max_length,truncation =True,return_tensors='pt').to(self.device)
        out_f = self.transformer(**(token_x))
        z = self.linear(out_f['last_hidden_state'])

        return z
