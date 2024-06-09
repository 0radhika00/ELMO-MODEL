import pandas as pd
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize
import numpy as np
import string
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import torch.nn.functional as F
from torchtext.vocab import build_vocab_from_iterator


UNKNOWN_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
trainLoss=[]

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

class Clean():
    def __init__(self):
        self.dict_word2idx={}
        self.ind=1
        self.list_of_list=[]
        # self.class_inst=class_inst

    def read_cvs(self,file,type_):
        df = pd.read_csv(file)
        if type_=='train':
            inputs = df['Description']
            classes =df['Class Index']
        else:
            inputs = df['Description']
            classes =df['Class Index']
        # if(type_=='train'):
        #     self.create_dic(df['Description'])#.head(15000))
        return self.data(inputs,classes)

    def data(self,input,cls):
        input_list=[]
        
        for i in input:
            
                input_list.append(self.clean(i))
        
        return list(zip(input_list, cls))
  
#     def create_dic(self,data):
        
#         self.dict_word2idx['<unk>']=0
# #         self.dict_word2idx['s>']=0
# #         self.dict_word2idx['</s>']=1
#         for sen in data:
#             filtered_sentence=self.clean(sen)
            
#             for word in filtered_sentence:
#                 if word not in self.dict_word2idx:
#                     self.dict_word2idx[word]=self.ind
#                     self.ind+=1

    def clean(self,text):
        # sen = number_regex.sub('num', text)
        sen=text.lower()
        
        punctuation_set = set(string.punctuation)
        tokens = word_tokenize(sen)
        # filtered_sentence = [word for word in tokens if word not in english_stopwords]
        filtered_sentence = [char for char in tokens if char not in punctuation_set]
        # print(filtered_sentence)
        return filtered_sentence
    
    #Dataset Creation
  

class NewsDataset(Dataset):
    def __init__(self, data,vocabulary):
        self.sentences=[i[0] for i in data ]
        self.labels=[i[1] for i in data]
        self.count=0
        if vocabulary is None:
                self.vocabulary = build_vocab_from_iterator(self.sentences, specials=[UNKNOWN_TOKEN, PAD_TOKEN]) # use min_freq for handling unkown words better
                self.vocabulary.set_default_index(self.vocabulary[UNKNOWN_TOKEN])
        else:
            # if vocabulary provided use that
                self.vocabulary = vocabulary
    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, index : int):
        # print(index)
        sen=self.sentences[index]
        # print(sen)
        l=[]
        for i in sen:
            if i in self.vocabulary:
                a=self.vocabulary[i]
            else:
                a=0
            # a=self.vocabulary.get(i, 0)
            l.append(a)
        return torch.tensor(l), torch.tensor(self.labels[index])

    def collate(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Given a list of datapoints, batch them together"""
        # print(batch)
        sentence = [i[0] for i in batch]
        labels = [i[1] for i in batch]
        padded_sentences = pad_sequence(sentence, batch_first=True) # pad sentences with pad token id
        # padded_labels = pad_sequence(labels, batch_first=True, padding_value=torch.tensor(0)) # pad labels with 0 because pad token cannot be entities

        return padded_sentences, torch.stack(labels)
    
# create a model
class Classification(nn.Module):
    def __init__(self,  in_dim,hidden_dim,output):
        super(Classification,self).__init__()
        self.hidden_dim = hidden_dim
    # define the layers
        self.lstm = nn.LSTM(in_dim, hidden_dim,batch_first=True)
        self.fc=   nn.Linear(hidden_dim, output)

        self.lbd0 = nn.Parameter(torch.Tensor([0.33]), requires_grad=False)
        self.lbd1 = nn.Parameter(torch.Tensor([0.33]), requires_grad=False)
        self.lbd2 = nn.Parameter(torch.Tensor([0.33]), requires_grad=False)

    def forward(self,f0_b0,f1_b1,f2_b2):
#         print(f0_b0.shape)
#         final_emb=self.lbd0*f0_b0+self.lbd1*f1_b1+self.lbd2*f2_b2
        final_emb=torch.cat((f0_b0,f1_b1,f2_b2),dim=2)
#         print(final_emb)
        out, _=self.lstm(final_emb)
#         print(out)
        out=self.fc(out[:,-1])
        return out



  

class Classify():
    def __init__(self) -> None:
        pass
    # Create a model training loop
    def train_model(self,num_epochs,train_data,learning_rate,forward_model,backward_model):
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        
        for epoch in tqdm(range(num_epochs)):
            print("Epoch:",epoch+1)
            ## TRAINING STEP
            model.train()
            # train
            running_loss=0
#             pbar=tqdm(train_data)
            for inputs,labels in train_data:
#                 inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                _,(e0,f0,f1)=forward_model(inputs)
                _,(e_b0,b0,b1)=backward_model(torch.flip(inputs, dims=[1]))
                f0_b0=  torch.cat((e0, e_b0), dim=2)
                f1_b1=  torch.cat((f0, b0), dim=2)
                f2_b2=  torch.cat((b1, f1), dim=2)
                outputs = model(f0_b0,f1_b1,f2_b2)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
#                 pbar.set_description(f'Training Loss: {running_loss / len(train_data.dataset):.6f}')
            _loss=running_loss/len(train_data)
            print(f"Train Loss: {_loss}")

def test(trained_model,test_data):
    
        c=0
        t=0
        pred_list=[]
        with torch.no_grad():
            for inputs,target in test_data:
                inputs,target=inputs.to(device),target.to(device)
                _,(e0,f0,f1)=forward_model(inputs)
                _,(e_b0,b0,b1)=backward_model(torch.flip(inputs, dims=[1]))
                f0_b0=  torch.cat((e0, e_b0), dim=2)
                f1_b1=  torch.cat((f0, b0), dim=2)
                f2_b2=  torch.cat((b1, f1), dim=2)
                outputs = model(f0_b0,f1_b1,f2_b2)
                _,pred=torch.max(outputs,1)
                pred_list.append(pred)
                t+=target.size(0)
                c+=(pred==target).sum().item()
        test_accuracy=c/t
        print('Test accuracy:',test_accuracy)

    
def evaluate_model(trained_model, data_loader):
    trained_model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs,targets=inputs.to(device),targets.to(device)
            _,(e0,f0,f1)=forward_model(inputs)
            _,(e_b0,b0,b1)=backward_model(torch.flip(inputs, dims=[1]))
            f0_b0=  torch.cat((e0, e_b0), dim=2)
            f1_b1=  torch.cat((f0, b0), dim=2)
            f2_b2=  torch.cat((b1, f1), dim=2)
            outputs = model(f0_b0,f1_b1,f2_b2)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.tolist())
            true_labels.extend(targets.tolist())

    accuracy = accuracy_score(true_labels, predictions)
    recall_micro = recall_score(true_labels, predictions, average='micro')
    recall_macro = recall_score(true_labels, predictions, average='macro')
    f1_micro = f1_score(true_labels, predictions, average='micro')
    f1_macro = f1_score(true_labels, predictions, average='macro')
    confusion_mat = confusion_matrix(true_labels, predictions)

    return accuracy, recall_micro, recall_macro, f1_micro, f1_macro, confusion_mat
 

print('Start')
# svd=SVD()
classification=Clean()


train_file='/kaggle/input/anlp-2/train.csv'
test_file='/kaggle/input/anlp-2/test.csv'
train_data=classification.read_cvs(train_file,'train')
test_data = classification.read_cvs(test_file,'test')


# print(svdcls.dict_word2idx)
print('Dataset')
train_dataset=NewsDataset(train_data,None)
test_dataset = NewsDataset(test_data,train_dataset.vocabulary)

print('Dataloader')
train_dataloader = DataLoader(train_dataset,batch_size=32, shuffle=True,collate_fn=train_dataset.collate)
test_dataloader = DataLoader(test_dataset,batch_size=32, shuffle=True,collate_fn=test_dataset.collate)

print('models')


forward_model= torch.load("/kaggle/input/modelsfinal/pytorch/1/1/forward_model.pt")
backward_model= torch.load("/kaggle/input/modelsfinal/pytorch/1/1/backward_model.pt")

# in_dim=600
in_dim=1800 #for function learning
output_size=5
lr=0.001
num_epoch=5

print('Classification Model')
model=Classification(in_dim,300,output_size).to(device)
lstm=Classify()
lstm.train_model(num_epoch,train_dataloader,lr,forward_model,backward_model)
print('Testing')
test(model,test_dataloader)

dev_accuracy, dev_recall_micro, dev_recall_macro, dev_f1_micro, dev_f1_macro, dev_confusion_mat = evaluate_model(model, train_dataloader)
print("Dev Set Evaluation:")
print("Accuracy:", dev_accuracy)
print("Recall (Micro):", dev_recall_micro)
print("Recall (Macro):", dev_recall_macro)
print("F1 Score (Micro):", dev_f1_micro)
print("F1 Score (Macro):", dev_f1_macro)
print("Confusion Matrix:")
print(dev_confusion_mat)

test_accuracy, test_recall_micro, test_recall_macro, test_f1_micro, test_f1_macro, test_confusion_mat = evaluate_model(model, test_dataloader)
print("\nTest Set Evaluation:")
print("Accuracy:", test_accuracy)
print("Recall (Micro):", test_recall_micro)
print("Recall (Macro):", test_recall_macro)
print("F1 Score (Micro):", test_f1_micro)
print("Confusion Matrix:")
print(test_confusion_mat)

torch.save(model, "classification_model.pt")