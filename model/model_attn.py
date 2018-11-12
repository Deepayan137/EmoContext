import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class SimpleLSTM(nn.Module):
    def __init__(self, nIn, nHidden):
        super(SimpleLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)

    def forward(self, input_):
        recurrent, (hidden, c) = self.rnn(input_)
        T, b, h = recurrent.size()
        return recurrent

class SimpleLinear(nn.Module):
    def __init__(self, nIn, nOut):
        super(SimpleLinear, self).__init__()
        self.linear = nn.Linear(nIn, nOut)

    def forward(self, x):
        timesteps, batch_size = x.size(0), x.size(1)
        x = x.view(batch_size*timesteps, -1)
        x = self.linear(x)
        x = x.view(timesteps, batch_size, -1)
        return x

class CNN_Embeddings(nn.Module):
    def __init__(self, nIn):
        super(CNN_Embeddings, self).__init__()

        Ci = 1
        Co = 100
        Ks = [3, 4, 5]

        self.conv13 = nn.Conv2d(Ci, Co, (3, nIn))
        self.conv14 = nn.Conv2d(Ci, Co, (4, nIn))
        self.conv15 = nn.Conv2d(Ci, Co, (5, nIn))

        self.dropout = nn.Dropout(0.5)


    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = x.unsqueeze(1)
        x1 = self.conv_and_pool(x, self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x, self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x, self.conv15) #(N,Co)
        z = torch.cat((x1, x2, x3), 1)  # (N,len(Ks)*Co)
        z = self.dropout(z)

        return z

class RNN_Embeddings(nn.Module):
    def __init__(self, nIn, nHidden, depth, span):
        super(RNN_Embeddings, self).__init__()
        self.fc_in = nn.Linear(nIn, nHidden*2)
        self.hidden_layers = [SimpleLSTM(nHidden*2, nHidden)for i in range(depth)]
        self.rnn = nn.Sequential(self.fc_in, *self.hidden_layers)
        self.attn = nn.Linear(nHidden*2, span)

    def attn_net(self, lstm_output):
        B,T, H = lstm_output.size()
        attn_weight_matrix = self.attn(lstm_output)
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
        out = F.softmax(attn_weight_matrix)
        return out

    def forward(self, x):
    # input_ = input_.permute(2, 1, 0).contiguous()
        x = x.permute(1, 0, 2) # t, b, h
        x = self.rnn(x)
        x = x.permute(1, 0, 2) # b, t, h
        attn_weight_matrix = self.attn_net(x) # b, 1, t
        sent_embedding = torch.bmm(attn_weight_matrix, x)
        sent_embedding = sent_embedding.squeeze(1)
        return sent_embedding
class RNN_attn(nn.Module):
    def __init__(self, nIn, nHidden, nOut, depth):
        super(RNN_attn, self).__init__()
        span = 10
        self.sent_embedding = RNN_Embeddings(nIn, nHidden, depth, span)
        self.fc_out = nn.Linear(nHidden*2*3*span, 4)

    def forward(self, x):
        rnn_feat = []
        for i in range(len(x)):
            hidden_matrix_rnn = self.sent_embedding(x[i])
            rnn_feat.append(hidden_matrix_rnn)
        rnn_out = torch.cat(rnn_feat, 1)
        rnn_out = rnn_out.view(rnn_out.size(0), -1)
        out  = self.fc_out(rnn_out)
        return out   
class WordEncoder(nn.Module):
    def __init__(self, nIn, nHidden, nOut, depth):
        super(WordEncoder, self).__init__()
        span = 5
        self.sent_embedding = RNN_Embeddings(nIn, nHidden, depth, span)
        self.cnn_embedding = CNN_Embeddings(nIn)
        self.fc_out_1 = nn.Linear(8580,1000)
        self.fc_out_2 = nn.Linear(1000, nOut)


    def forward(self, x):
        # input_ = input_.permute(2, 1, 0).contiguous()
        
        rnn_feat, cnn_feat = [], []
        for i in range(len(x)):
            hidden_matrix_rnn = self.sent_embedding(x[i])
            # pdb.set_trace()
            hidden_matrix_cnn = self.cnn_embedding(x[i])
            hidden_matrix_cnn = hidden_matrix_cnn.unsqueeze(1)
            cnn_feat.append(hidden_matrix_cnn)
            rnn_feat.append(hidden_matrix_rnn)
        cnn_out = torch.cat(cnn_feat, 1) # 32, 3, 300
        cnn_out = cnn_out.view(cnn_out.size(0), -1)
        rnn_out = torch.cat(rnn_feat, 1) # 32, 15, 512
        rnn_out = rnn_out.view(rnn_out.size(0), -1)
        out = torch.cat((cnn_out, rnn_out),1)
        fc_out = self.fc_out_1(out)
        fc_out = self.fc_out_2(fc_out)
        #==============================
        # cnn_out = cnn_out.permute(1, 0, 2)
        # fc_out = self.fc_out_2(sent_embedding)
        #=======================================
        # out = torch.cat(rnn_feat, 1)
        # out = out.view(out.size(0), -1)
        # # pdb.set_trace()
        # fc_out = self.fc_out_1(out)
        # fc_out = self.fc_out_2(fc_out)
        return fc_out