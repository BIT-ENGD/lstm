# https://blog.csdn.net/dendi_hust/article/details/94435919
# https://www.cnblogs.com/cxq1126/p/13504437.html

# 完整代码：https://blog.csdn.net/qq_34523665/article/details/105664659?

# bilsmt 详解  https://blog.csdn.net/baidu_38963740/article/details/117197619

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

'''
Attention计算
将BILSTM网络输出的结果（shape：[batch_size, time_step, hidden_dims * num_directions(=2)]）拆成两个大小为[batch_size, time_step, hidden_dims]的Tensor；
将第一步拆出的两个Tensor进行相加运算得到h（shape：[batch_size, time_step, hidden_dims]）；
将BILSTM网络最后一个隐状态（shape：[batch_size, num_layers * num_directions, hidden_dims]）在第二维度进行求和，得到新的lstm_hidden（shape：[batch_size, hidden_dims]）；
将lstm_hidden的维度从[batch_size, n_hidden]扩展到[batch_size, 1, hidden_dims]；
使用slef.atten_layer(h)获得用于后续计算权重的向量atten_w（shape：[batch_size, 1, hidden_dims]）；
将h进行tanh激活，得到m（shape：[batch_size, time_step, hidden_dims]）；
使用torch.bmm(atten_w, m.transpose(1, 2)) 得到atten_context（shape：[batch_size, 1, time_step]）；
将atten_context使用F.softmax(atten_context, dim=-1)进行归一化，得到基于上下文权重的softmax_w（shape：[batch_size, 1, time_step]）；
使用torch.bmm(softmax_w, h)得到基于权重的BILSTM输出context（shape：[batch_size, 1, hidden_dims]）;
将context的第二维度消掉，得到result（shape：[batch_size, hidden_dims]） ;
返回result；

'''

dtype=torch.FloatTensor

embedding_dim=3
n_hidden=5
num_classes=2
sentences=["i love you","he loves me","she likes baseball","i hate you","sorry for that","this is awful"]
labels=[1,1,1,0,0,0]

word_list=" ".join(sentences).split()
word_list=list(set(word_list))
word_dict={w:i for i,w in enumerate(word_list)}
vocab_size=len(word_dict)

inputs=[]
for sen in sentences:
   inputs.append([word_dict[n] for n in sen.split()])

targets=[]

for out in labels:
   targets.append(out)

input_batch=torch.LongTensor(inputs)
target_batch=torch.LongTensor(targets)

class BiLSTM_Attention(nn.Module):
   def __init__(self):
       super(BiLSTM_Attention,self).__init__()
       self.embedding = nn.Embedding(vocab_size,embedding_dim)
       self.lstm=nn.LSTM(embedding_dim,n_hidden,bidirectional=True)
       self.out = nn.Linear(n_hidden*2,num_classes)
   
   def attention_net(self,lstm_output,final_state):
      hidden = final_state.view(-1,n_hidden*2,1)
      attn_weights= torch.bmm(lstm_output,hidden).squeeze(2)
      soft_attn_weights = F.softmax(attn_weights,1)
      context=torch.bmm(lstm_output.transpose(1,2),soft_attn_weights.unsqueeze(2)).squeeze(2)
      return context,soft_attn_weights

   def forward(self,X):
      input=self.embedding(X)  # 6，3 输入数据，输出 6,3,3
      input=input.permute(1,0,2)  # [3,6,6]
      hidden_state=torch.zeros(1*2,len(X),n_hidden)
      cell_state=torch.zeros(1*2,len(X),n_hidden)
      output,(final_hidden_state, final_cell_state) = self.lstm(input,(hidden_state,cell_state))
      output=output.permute(1,0,2)
      attn_output, attention= self.attention_net(output,final_hidden_state)
      return self.out(attn_output),attention






model=BiLSTM_Attention()
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)
'''

'''
for epoch in range(5000):
   
   output,attention=model(input_batch)
   loss=criterion(output,target_batch)
   if  (epoch+1)%1000 ==0:
      print("Epoch:","%04d"%(epoch+1),"cost=",'{:.6f}'.format(loss))
   optimizer.zero_grad()
   loss.backward()
   optimizer.step()


test_text="i love that"
tests=[[word_dict[n] for n in test_text.split()]]
test_batch=torch.LongTensor(tests)
predict, _ =model(test_batch)
predict=predict.data.max(1,keepdim=True)[1]
if predict[0][0] == 0:
   print(test_text," is Bad mean...")
else:
   print(test_text," is Good Mean!!")


'''
# Test
test_text = 'sorry hate you'
tests = [[word_dict[n] for n in test_text.split()]]
test_batch = torch.LongTensor(tests)
# Predict
predict, _ = model(test_batch)
predict = predict.data.max(1, keepdim=True)[1]
if predict[0][0] == 0:
    print(test_text,"is Bad Mean...")
else:
    print(test_text,"is Good Mean!!")
'''

fig = plt.figure(figsize=(6, 3)) # [batch_size, n_step]
ax = fig.add_subplot(1, 1, 1)
ax.matshow(attention.detach().numpy(), cmap='viridis')
ax.set_xticklabels(['']+['first_word', 'second_word', 'third_word'], fontdict={'fontsize': 14}, rotation=90)
ax.set_yticklabels(['']+['batch_1', 'batch_2', 'batch_3', 'batch_4', 'batch_5', 'batch_6'], fontdict={'fontsize': 14})
plt.show()
