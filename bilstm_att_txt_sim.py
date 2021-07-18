from typing import Iterator
import torch 
import torch.nn as nn 
from  torch.utils.data import Dataset,DataLoader
import torchtext.vocab as vocab
from torchtext.legacy.data import Field,TabularDataset, Example,BucketIterator
import torch.functional as F
import jieba 
import gensim
import math

# https://blog.csdn.net/huanxingchen1/article/details/107185861    torchtext 使用方法

# https://dzlab.github.io/dltips/en/pytorch/torchtext-datasets/  torchtext 数据集使用方法


# lstm 详解 https://zhuanlan.zhihu.com/p/79064602



# 参考代码： https://codechina.csdn.net/mirrors/WHLYA/text-classification/-/blob/master/text%20classification/LSTM+Attention.ipynb
'''
lstm的 Hiddien_size 其实就是W 权重的维数

input_size 即输入参数的维数，比如embedding_dim （嵌入变量的维数）
batch_first: 第一维是否是 batch_size
输出维度为hidden_size, 双向bilstm为hidden_size*2


默认输入数据为 input,(h_0,c_0)
input: 输入数据，即上面例子中的一个句子（或者一个batch的句子），其维度形状为 (seq_len, batch, input_size), input_size为输入变量的维度，比如embedding_dim
h_0：维度形状为 (num_layers * num_directions, batch, hidden_size):
c_0： 维度形状为 (num_layers * num_directions, batch, hidden_size),各参数含义和h_0类似。
当然，如果你没有传入(h_0, c_0)，那么这两个参数会默认设置为0。


输出数据 output,(Hn,Cn)
output: [seq_len,bacth,num_direction*hidden_size]

Hn (num_layers*num_direction,batch,hidden_size)
Cn (num_layers* num_directions, batch, hidden_size)

'''

'''
def forward(self,x):
    print(x.shape)
    text_emb = self.word_embeddings(x)
    print(text_emb.shape)
    lstm_out, lstm_hidden = self.lstm(text_emb)
    lstm_out = lstm_out[:,-1,:]
    print(lstm_out.shape)
    output = self.dense(drop_out)
    return output

'''

embedding_dim =128
n_hidden=5
num_classes=120
batch_size=100
EPOCH=500

DATASETDIR="H:\\dataset\\"
W2V_TXT_FILE="w2v\\baike_26g_news_13g_novel_229g.txt"
W2V_BIN_FILE="w2v\\baike_26g_news_13g_novel_229g.bin"
CACHE_DIR="w2v\\cache"


''' 
DATA_DIR=DATASETDIR+"LCQMC\\data\\"
TRAIN_DATA="train.tsv"
DEV_DATA="dev.tsv"
TEST_DATA="test.tsv"
'''

DATA_DIR=DATASETDIR+"iflytek_public\\"
TRAIN_DATA="train.json"
DEV_DATA="dev.json"
TEST_DATA="test.json"

STOPFILE=DATASETDIR+"stopwords.txt"


SENTENCE_LEN=50
# prepare stoplist

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


STOPLIST=['在','上','的','地','得','就是','是','了','和','就','还','一直','中','让','提前','如','！','？','：','；','－','只能','以','只','，','。','！','、','——','《','》','（','）','到','从','之','【', '】','/','(',')','-','◆','才','最','已','□','却','更']
with open(STOPFILE,encoding="utf-8") as f:
        for item in f:
            line=item.replace('\n', '').replace('\r', '').strip()
            STOPLIST.append(line)
def is_stop_word(word):
    if word in STOPLIST:
        return True
    else:
        return False


# bin to txt 转换代码
#model = gensim.models.KeyedVectors.load_word2vec_format(DATASETDIR+W2V_BIN_FILE,binary=True)
#model.save_word2vec_format(DATASETDIR+W2V_TXT_FILE)

def tokenizer(text):    # 可以自己定义分词器，比如jieba分词。也可以在里面添加数据清洗工作     "分词操作，可以用jieba"
     return [wd for wd in jieba.cut(text, cut_all=False) if not is_stop_word(wd)]
 
         


TEXT=Field(sequential=True, tokenize=tokenizer,lower=True,fix_length=200)
LABEL=Field(sequential=False, use_vocab=False,preprocessing=int)

vectors=vocab.Vectors(name=DATASETDIR+W2V_TXT_FILE,cache=DATASETDIR+CACHE_DIR)
weights=torch.FloatTensor(vectors.vectors)
embed_dim=weights.size(1)
vocnum=weights.size(0)
print(embed_dim,vocnum)

#fields=[("first",TEXT),("second",TEXT),("similarity",LABEL)]
#train, test = TabularDataset.splits(path=DATA_DIR,format="tsv",train=TRAIN_DATA,test=TEST_DATA,skip_header=False, fields=fields)


fields_train = {
  'label': ('label', LABEL),
  #'label_des': ('label_des', TEXT),
  'sentence': ('sentence', TEXT) 
}

fields_test = {
   'sentence': ('sentence', TEXT) 
}

train= TabularDataset.splits(path=DATA_DIR,format="json",train=TRAIN_DATA,skip_header=False, fields=fields_train)[0]
test= TabularDataset.splits(path=DATA_DIR,format="json",test=TEST_DATA,skip_header=False, fields=fields_test)[0]
TEXT.build_vocab(train,max_size=50000)   #构建词表
LABEL.build_vocab(train) # 
TEXT.vocab.set_vectors(vectors.stoi,vectors.vectors,vectors.dim)  #替换向量为word2vec
embedding =nn.Embedding.from_pretrained(torch.FloatTensor(TEXT.vocab.vectors))  #准备训练用向量
index=vectors.stoi["中国"]
china_vec=vectors.vectors[index]

# for a padding word, its index is 1.
def create_embed(TEXT,sentence):
    ori_list=[TEXT.vocab.stoi[w] for w in sentence]
    if len(ori_list) > SENTENCE_LEN :
        return torch.IntTensor(ori_list[:SENTENCE_LEN])
    else:
        return torch.IntTensor(ori_list + (SENTENCE_LEN-len(ori_list))*[1])

china_vec2=embedding(create_embed(TEXT,["中国","人类"]))
print(china_vec)
print(china_vec2)

class BiLSTM_Attention(nn.Module):
   def __init__(self,embed):
       super(BiLSTM_Attention,self).__init__()
       #self.embedding = nn.Embedding(vocab_size,embedding_dim)
       self.embedding=embed
       self.lstm=nn.LSTM(embedding_dim,n_hidden,bidirectional=True) # bidirection
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
      return self.out(attn_output),attention  # FC 全连接网络 降维用


class BiLSTM_AttentionEx(nn.Module):
    def __init__(self, embedding,embedding_dim, hidden_dim, n_layers):

        super(BiLSTM_Attention, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = embedding
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=True, dropout=0.5)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.5)
    #x,query：[batch, seq_len, hidden_dim*2]
    def attention_net(self, x, query, mask=None):      #软性注意力机制（key=value=x）

        d_k = query.size(-1)                                              #d_k为query的维度
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)  #打分机制  scores:[batch, seq_len, seq_len]

        p_attn = F.softmax(scores, dim = -1)                              #对最后一个维度归一化得分
        context = torch.matmul(p_attn, x).sum(1)       #对权重化的x求和，[batch, seq_len, hidden_dim*2]->[batch, hidden_dim*2]
        return context, p_attn

    def forward(self, x):
        embedding = self.dropout(self.embedding(x))       #[seq_len, batch, embedding_dim]

        # output: [seq_len, batch, hidden_dim*2]     hidden/cell: [n_layers*2, batch, hidden_dim]
        output, (final_hidden_state, final_cell_state) = self.rnn(embedding)
        output = output.permute(1, 0, 2)                  #[batch, seq_len, hidden_dim*2]

        query = self.dropout(output)
        attn_output, attention = self.attention_net(output, query)       #和LSTM的不同就在于这一句
        logit = self.fc(attn_output)
        return logit


# training

# https://zhuanlan.zhihu.com/p/353795265
train_iter = BucketIterator.splits( (train),sort_key=lambda x: len(x.text), batch_size=(batch_size),device=device,sort_within_batch=False)
test_iter=Iterator(test, batch_size=64, device=device, sort=False, sort_within_batch=False, shuffle=False)
vocab_size=len(TEXT.vocab)
label_num = len(LABEL.vocab)

print(vocab_size, label_num)


