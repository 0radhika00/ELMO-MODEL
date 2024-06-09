import pandas as pd
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize
import numpy as np
import string
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from tqdm import tqdm
from torchtext.vocab import build_vocab_from_iterator, Vocab
import torch.nn.functional as F
import os

UNKNOWN_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class CleanData:
    def __init__(self) -> None:
   
   
       
       
        self.input_forward=[]
        self.input_backward=[]
        self.output_forward=[]
        self.output_backward=[]
        

    def read_cvs(self,file,type_of_file):
        df = pd.read_csv(file)
        column_data = df['Description']
        self.clean(column_data,type_of_file)



    def clean(self,data,type_of_file):
         
        
        #  print("Data length",len(data))
         for sen in data:
            sen=sen.lower()
            tokens = word_tokenize(sen)
            punctuation_set = set(string.punctuation)
            filtered_sentence = [char for char in tokens if char not in punctuation_set]
            
            reverse_sen=filtered_sentence[::-1]
            reverse_sen.append('<s>')
            filtered_sentence.insert(0,'<s>')
#             print(filtered_sentence,reverse_sen)
            self.input_forward.append(filtered_sentence)
            self.output_forward.append(filtered_sentence[1:])
            self.input_backward.append(reverse_sen)
            self.output_backward.append(reverse_sen[1:])
            
            
            # if type_of_file=='train':
            #     self.creation(tokens)
            

    # def creation(self,sen):
    #     #vocab creation
    #     for i in range(len(sen)):
    #         if sen[i] not in self.vocab:
    #            self.vocab[sen[i]]=self.index
    #            self.index=self.index+1 
               
  

class NewsDataset(Dataset):
    def __init__(self, data,vocabulary):
        self.word_context=[i[0] for i in data]
        self.labels=[i[1] for i in data]
        
        if vocabulary is None:
            self.vocabulary = build_vocab_from_iterator(self.word_context, specials=[UNKNOWN_TOKEN, PAD_TOKEN]) # use min_freq for handling unkown words better
            self.vocabulary.set_default_index(self.vocabulary[UNKNOWN_TOKEN])
        else:
        # if vocabulary provided use that
            self.vocabulary = vocabulary


    def __len__(self) -> int:
        return len(self.word_context)

    def __getitem__(self, index : int):
    # print(index)
        sen=self.word_context[index]
        lab=self.labels[index]
        l1=[self.vocabulary[i] for i in sen]
        l2=[self.vocabulary[i]  for i in lab]
       
        # print(l1,l2)
        return torch.tensor(l1),torch.tensor(l2)
        # return torch.tensor(self.vocabulary.lookup_indices(self.word_context[index])), torch.tensor(self.vocabulary.lookup_indices(self.labels[index]))

    def collate(self, batch) :
        """Given a list of datapoints, batch them together"""
        # print(batch)
        sentence = [i[0] for i in batch]
        labels = [i[1] for i in batch]
        padded_sentences = pad_sequence(sentence, batch_first=True,padding_value=self.vocabulary[PAD_TOKEN]) # pad sentences with pad token id
        padded_labels = pad_sequence(sentence, batch_first=True, padding_value=torch.tensor(0)) # pad labels with 0 because pad token cannot be entities

        return padded_sentences, padded_labels

    
    
class forward_backward(nn.Module):
    def __init__(self, vocab_size, embedding_dimension,hidden_dim,output):
        super(forward_backward,self).__init__()
        self.embedding=nn.Embedding(vocab_size,embedding_dimension)
        self.lstm1=nn.LSTM(embedding_dimension, embedding_dimension,1,batch_first=True)
        self.lstm2=nn.LSTM(embedding_dimension, embedding_dimension,1,batch_first=True)
        self.fc=nn.Linear(embedding_dimension,output)

    def forward(self,x):
        emb=self.embedding(x)
        out1 , _=self.lstm1(emb)
        out2 , _=self.lstm2(out1)
        # print(out2.shape)
        # out=F.relu(out2)
        # print(out.shape)
        out=self.fc(out2)
        # print(out.shape)
        return out,(emb,out1,out2)
    

print('Read File')
train_file='/kaggle/input/anlp-2/train.csv'
test_file='/kaggle/input/anlp-2/test.csv'

print('Clean Data')
clean=CleanData()
clean.read_cvs(train_file,'train')
forward_train=list(zip(clean.input_forward,clean.output_forward))
# print(forward_train[0])
backward_train=list(zip(clean.input_backward,clean.output_backward))



print('Dataset Creation')
train_dataset_forward=NewsDataset(forward_train,None)
# print(train_dataset_forward[0])
train_dataset_backward=NewsDataset(backward_train,train_dataset_forward.vocabulary)
print('size of vocab',len(train_dataset_forward.vocabulary))

print('Dataloader Creation')
train_dataloader_forward = DataLoader(train_dataset_forward,batch_size=50, shuffle=True,collate_fn=train_dataset_forward.collate)
train_dataloader_backward = DataLoader(train_dataset_backward,batch_size=50,shuffle=True ,collate_fn=train_dataset_backward.collate)
# for i in train_dataloader_forward:
#     print(i,i[0].shape,i[1].shape)
#     break


print('Defining Model')
emb_dim=300
hid_dim=300
output=len(train_dataset_forward.vocabulary)
forward_model=forward_backward(len(train_dataset_forward.vocabulary),emb_dim,hid_dim,output)
backward_model=forward_backward(len(train_dataset_forward.vocabulary),emb_dim,hid_dim,output)
forward_model=forward_model.to(device)
backward_model=backward_model.to(device)
epoch=1
criterion = nn.CrossEntropyLoss()
optimizer1 = torch.optim.Adam(forward_model.parameters(), lr=0.001)
optimizer2 = torch.optim.Adam(backward_model.parameters(),lr=0.001)

print('Training forward')
for e in range(epoch):
    print(f'Epoch {e+1}/{epoch}')
    forward_model.train()
    # backward_model.train()
    running_loss = 0.0
    pbar = tqdm(train_dataloader_forward)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        # print(inputs,targets,inputs.shape,targets.shape)
        # inputs = torch.stack(inputs)
        # targets = torch.stack(targets)
        optimizer1.zero_grad()
        outputs,_ = forward_model(inputs)
        # print(outputs.shape)
        loss = criterion(outputs.reshape(-1, outputs.size(2)), targets.reshape(-1))
        # loss=criterion(outputs,targets)
        loss.backward()
        optimizer1.step()
        running_loss+=loss.item()
     
        _loss=running_loss/len(train_dataloader_forward)
        print(f"Train Loss: {_loss}")
torch.save(forward_model, "forward_model.pt")

print('Training backward')
for e in range(epoch):
    print(f'Epoch {e+1}/{epoch}')
    # forward_model.train()
    backward_model.train()
    running_loss = 0.0
    pbar = tqdm(train_dataloader_backward)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer1.zero_grad()
        outputs,_ = forward_model(inputs)
        loss = criterion(outputs.reshape(-1, outputs.size(2)), targets.reshape(-1))
        loss.backward()
        optimizer2.step()
        running_loss+=loss.item()
     
        _loss=running_loss/len(train_dataloader_backward)
        print(f"Train Loss: {_loss}")

    
torch.save(backward_model,'backward_model.pt')
