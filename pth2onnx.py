
import torch
import torch.nn as nn
import math 
import torch.nn.functional as F
import onnx
MODEL_NAME="bilstm_att_txt.pth"
MODEL_ONNX_NAME="best_model_new.onnx"

num_classes =120
EPOCH_NUM=200
n_hidden=240

batch_size=2200

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

#https://www.codenong.com/cs107117759/
def ExportModel(model,input):
    torch.onnx.export(model,               # model being run
                sentence,                         # model input (or a tuple for multiple inputs)
                MODEL_ONNX_NAME,   # where to save the model (can be a file or file-like object)
                export_params=True,        # store the trained parameter weights inside the model file
                opset_version=12,          # the ONNX version to export the model to
                do_constant_folding=True,  # whether to execute constant folding for optimization
                input_names = ['embedding'],   # the model's input names
                output_names = ['dropout'], # the model's output names
                dynamic_axes={'embedding' : {0:"string_len",1:"batch_size"}    # variable lenght axes
                         }   
        )
newmodel=torch.load(MODEL_NAME)
sentence=torch.randint(low=0,high=22000,size=(300,1),device=torch.device("cuda"))
ExportModel(newmodel,sentence)
onnxmodel=onnx.load(MODEL_ONNX_NAME)
onnx.checker.check_model(onnxmodel)
onnx.helper.printable_graph(onnxmodel.graph)
