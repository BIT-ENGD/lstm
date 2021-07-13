import torch 
import torch.nn as nn 
from  torch.utils.data import Dataset,DataLoader
import gensim
import torchtext.vocab as vocab

from torchtext.legacy.data import Field,TabularDataset, Example,BucketIterator

import jieba 


# https://blog.csdn.net/huanxingchen1/article/details/107185861    torchtext 使用方法

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

DATASETDIR="H:\\dataset\\"
W2V_TXT_FILE="w2v\\baike_26g_news_13g_novel_229g.txt"
W2V_BIN_FILE="w2v\\baike_26g_news_13g_novel_229g.bin"
CACHE_DIR="w2v\\cache"



DATA_DIR=DATASETDIR+"LCQMC\\data\\"
TRAIN_DATA="train.tsv"
DEV_DATA="dev.tsv"
TEST_DATA="test.tsv"

STOPFILE=DATASETDIR+"stopwords.txt"

# prepare stoplist



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

fields=[("first",TEXT),("second",TEXT),("similarity",LABEL)]
train, test = TabularDataset.splits(path=DATA_DIR,format="tsv",train=TRAIN_DATA,test=TEST_DATA,skip_header=False, fields=fields)
TEXT.build_vocab(train,max_size=50000)   #构建词表
LABEL.build_vocab(train) # 
TEXT.vocab.set_vectors(vectors.stoi,vectors.vectors,vectors.dim)  #替换向量为word2vec
embedding =nn.Embedding.from_pretrained(torch.FloatTensor(TEXT.vocab.vectors))  #准备训练用向量
index=vectors.stoi["中国"]
china_vec=vectors.vectors[index]


def create_embed(vectors,sentence):
    return [ vectors.vectors[vectors.stoi[w]] for w in sentence]

china_vec2=create_embed(vectors,["中国","人类"])

print(china_vec)
print(china_vec2)



class BiLSTM_Attn(nn.Module):
    def __init__(self,input,embedding):
        super(BiLSTM_Attn,self).__init__()
        self.embedding =embedding
        self.X=input


#print(TEXT.vocab)
print(LABEL.vocab)