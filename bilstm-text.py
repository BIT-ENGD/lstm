import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import copy


# https://cloud.tencent.com/developer/article/1652786

# https://blog.csdn.net/weixin_41041772/article/details/88032093 

# https://blog.csdn.net/baidu_38963740/article/details/117197619  详解 


# 对输入 数据的理解 https://blog.csdn.net/yyb19951015/article/details/79740869?utm_source=blogxgwz8

#https://blog.csdn.net/qq_27318693/article/details/85642827

dtype =torch.FloatTensor

sentence=(
    'GitHub Actions makes it easy to automate all your software workflows '
    'from continuous integration and delivery to issue triage and more'
)

device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
word2idx={w: i for i ,w in enumerate(list(set(sentence.split())))} # 单词、索引查找 表
idx2word={i: w for i ,w in enumerate(list(set(sentence.split())))} #索引 、单词对照表

n_class=len(word2idx)   #19个词，输出 也是19个，类似分类操作
max_len=len(sentence.split())
n_hidden =8


def make_data(sentence):
    input_batch =[]
    target_batch=[]

    words=sentence.split()
    for i in range(max_len-1):
        input = [word2idx[n] for n in words[:(i+1)]]
        input = input +[0]* (max_len -len (input)) #前n个单词用索引 填充，其它 的位置 用0填充
        target =word2idx[words[i+1]]  #下一个词的索引值 
        tt=np.eye(n_class)[input]
        input_batch.append(tt)
        target_batch.append(target)
    
    return torch.Tensor(input_batch), torch.LongTensor(target_batch)

#input_batch: [max_len-1, max_len, n_class]
input_batch, target_batch =make_data(sentence)

dataset=Data.TensorDataset(input_batch,target_batch)
loader=Data.DataLoader(dataset,16, True)# 16为batch_size



class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM,self).__init__()
        self.lstm =nn.LSTM(input_size=n_class,hidden_size=n_hidden, bidirectional=True)
        self.fc=nn.Linear(n_hidden*2,n_class)

    def forward(self,X):
        #X [batch-size, max_len, n_class]  #批大小，序列长度，输入 的数据的维度（这儿跟类数相同）。原始的输入 是由[batch_size,input_dim,time_step] 组成。 
        batch_size=X.shape[0]
        input = X.transpose(0,1).to(device) # input: [manx_len, batch_size, n_class]

        hidden_state=torch.randn(1*2,batch_size,n_hidden).to(device) #[ number_layers (=1) * num_directions(=2), batch_size, n_hidden]
        cell_state=torch.randn(1*2, batch_size,n_hidden).to(device)  #[ number_layers (=1) * num_directions(=2), batch_size, n_hidden]
        
        outputs, (ho,co) = self.lstm(input,(hidden_state,cell_state))
        outputs = outputs[-1] #[ batch_sze, n_hidden*2]
        model=self.fc(outputs)  #model: [batch_size, n_class]
        return model

model = BiLSTM().to(device)
criterion=nn.CrossEntropyLoss().to(device)
optimizer=optim.Adam(model.parameters(),lr=0.001)


# training 
loss_min=1
for epoch in range(20000):
    for x,y in loader:
        x=x.to(device)
        y=y.to(device)
        pred=model(x).to(device)
        loss=criterion(pred,y)
        if (epoch +1 )% 1000 ==0:
            print("Epoch:","%04d" % (epoch+1), "loss=","{:.6f}".format(loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if  epoch > 8000 and loss < loss_min:
        loss_min=loss 
        best_model= copy.deepcopy(model)
        


print("mini Loss:", loss_min.item())
model_out=best_model(input_batch)
predict=model_out.data.max(1,keepdim=True)[1] #返回最大值的索引 
print(sentence)
print([idx2word[n.item()] for n in predict.squeeze()])


