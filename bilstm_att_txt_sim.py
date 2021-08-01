import torch 
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
import torchtext.vocab as vocab
from torchtext.legacy.data import Field,TabularDataset, Example,BucketIterator,Iterator
from torch.utils.tensorboard import SummaryWriter
import gensim
import torchsummary
import jieba 
import math 
import copy
import torch.onnx

# https://blog.csdn.net/huanxingchen1/article/details/107185861    torchtext 使用方法

# https://dzlab.github.io/dltips/en/pytorch/torchtext-datasets/  torchtext 数据集使用方法

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

DATASETDIR="f:\\dataset\\"
W2V_TXT_FILE="w2v\\baike_26g_news_13g_novel_229g.txt"
W2V_BIN_FILE="w2v\\baike_26g_news_13g_novel_229g.bin"
CACHE_DIR="w2v\\cache"


num_classes =120
EPOCH_NUM=200
n_hidden=240

batch_size=2200

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

MODEL_NAME="bilstm_att_txt.pth"

SENTENCE_LEN=50
# prepare stoplist


device=torch.device("cuda"  if torch.cuda.is_available() else "cpu")


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
LABEL=Field(sequential=False, use_vocab=False)

vectors=vocab.Vectors(name=DATASETDIR+W2V_TXT_FILE,cache=DATASETDIR+CACHE_DIR)
weights=torch.FloatTensor(vectors.vectors)
embed_dim=weights.size(1)
vocnum=weights.size(0)
print(embed_dim,vocnum)

#fields=[("first",TEXT),("second",TEXT),("similarity",LABEL)]
#train, test = TabularDataset.splits(path=DATA_DIR,format="tsv",train=TRAIN_DATA,test=TEST_DATA,skip_header=False, fields=fields)


fields_train = {
  'label': ('label', LABEL),
  #'label_des': ('label_des', None),
  'sentence': ('sentence', TEXT) 
}


fields_test = {
  #'label_des': ('label_des', None),
  'sentence': ('sentence', TEXT) 
}

train = TabularDataset.splits(path=DATA_DIR,format="json",train=TRAIN_DATA,skip_header=False, fields=fields_train)[0]
test = TabularDataset.splits(path=DATA_DIR,format="json",test=TEST_DATA,skip_header=False, fields=fields_test)[0]
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
'''
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

'''
class BiLSTM_AttentionEx(nn.Module):
    def __init__(self, embedding,embedding_dim, hidden_dim, n_layers):

        super(BiLSTM_AttentionEx, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = embedding
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=True, dropout=0.5)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.5) # 需要调用train
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
#https://blog.csdn.net/liu_chengwei/article/details/115299789  dataset.iterator等的 使用
train_iter = BucketIterator( (train),sort_key=lambda x: len(x.text), batch_size=(batch_size),device=device,sort_within_batch=False)  #BucketIterator.splits
test_iter=Iterator(test, batch_size=64, device=device, sort=False, sort_within_batch=False, train=False,shuffle=False)   # 对  测试集，train要为false
vocab_size=len(TEXT.vocab)
label_num = len(LABEL.vocab)


model=BiLSTM_AttentionEx(embedding,embed_dim,n_hidden,2).to(device)


#print(vocab_size, label_num)

# https://www.cnblogs.com/tangzz/p/14598268.html

LR=1e-2
optimizer=optim.Adam(model.parameters(),lr=LR)
criterion =F.cross_entropy
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

# 定义tensorboard日志的输出目录
writer = SummaryWriter("runs/lstm")
model.train() # 对于有dropout和BN层的，需要调用 
    
loss_epoch=0.0
loss_min=1.0
correct_num=0.0
total_num=0.0
best_model=model
is_best=False
sentence="'"
for epoch in range(EPOCH_NUM):

  #  scheduler.step()

    for i, batch in enumerate(train_iter):
       
        optimizer.zero_grad()
        label,sentence=batch.label,batch.sentence
        pred=model(sentence)  #推理
        loss =criterion(pred,label) #计算loss 
        loss.backward()  #反向传递
        optimizer.step()  # 更新参数

        if loss < loss_min:
            loss_min=loss 
            best_model= copy.deepcopy(model)
            is_best=True


print("done loss_min:",loss_min)

#https://www.codenong.com/cs107117759/
def ExportModel(model):
    torch.onnx.export(model,               # model being run
                 sentence,                         # model input (or a tuple for multiple inputs)
                  "best_model.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['embedding'],   # the model's input names
                  output_names = ['dropout'], # the model's output names
                  dynamic_axes={'embedding' : [0,1],     # variable lenght axes
                                'dropout' : [0,1]})
if is_best:
    torch.save(best_model,MODEL_NAME)
    newmodel=torch.load(MODEL_NAME)
    print("Get a best model.")
    ExportModel(best_model)
