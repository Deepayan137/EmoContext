import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        return output

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

class CNN_Text(nn.Module):
    
    def __init__(self):
        super(CNN_Text, self).__init__()
        
        D = 25
        C = 4
        Ci = 1
        Co = 100
        Ks = [3, 4, 5]

        # self.embed = nn.Embedding(V, D)
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        # self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
    
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(len(Ks)*Co, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        # x = self.embed(x)  # (N, W, D)
        # pdb.set_trace() 
        x = x.permute(1, 0, 2) # N*W*D
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        # x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        # x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        # x = torch.cat(x, 1)

        
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        
        x = self.dropout(x)  # (N, len(Ks)*Co)
        
        logit = self.fc1(x)  # (N, C)
        return logit

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        nIn = 25
        nOut = 100
        self.module = nn.Sequential(nn.Conv1d(nIn, nOut, 
                                    kernel_size=3, 
                                    stride=1, 
                                    padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool1d(156)
                                    )
        self.fc = nn.Linear(100, 4)
        
    def forward(self, x):
        input_size = x.size() # T*B*D
        x_orig = x
        x = x.permute(1, 2, 0) # B*D*T
        output = self.module(x).squeeze(2)
        output = self.fc(output.view(-1, 100))
        return output

class RCNN(nn.Module):
    def __init__(self, nIn, nHidden, nOut, seq_len, filters):
        super(RCNN, self).__init__()
        nCh = 1
        self.fc_in = nn.Linear(nIn, 2*nHidden)
        # self.rnn = nn.LSTM(2*nHidden, nHidden, bidirectional=True)
        self.hidden_layers = [SimpleLSTM(nHidden*2, nHidden)for i in range(2)]
        self.rnn = nn.Sequential(*self.hidden_layers)
        self.module = nn.Sequential(nn.Conv2d(nCh, filters, 
                                    kernel_size=3, 
                                    stride=1, 
                                    padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2)
                                    )
        
        seqlen_pool2d = seq_len//2
        self.fc_out = nn.Linear(nHidden*seqlen_pool2d*filters, 4)

    def forward(self, x):
        T, B, D = x.size()
        x = x.view(-1, D)
        x = self.fc_in(x) # T*hidden
        x = x.view(T, B,-1) # T*B*hidden
        # recurrent, (hidden, c)  = self.rnn(x) #T*B*hidden
        recurrent  = self.rnn(x) #T*B*hidden
        recurrent = recurrent.unsqueeze(1) # B*Ch*D*T
        conv = self.module(recurrent)
        B, Ch, H, W = conv.size()
        # pdb.set_trace()
        conv = conv.view(-1, Ch*H*W)
        output = self.fc_out(conv)
        return output
        

class EmoNet(nn.Module):
    def __init__(self, nIn, nHidden, nOut, depth):
        super(EmoNet, self).__init__()
        self.fc_in = SimpleLinear(nIn, nHidden*2)
        self.hidden_layers = [SimpleLSTM(nHidden*2, nHidden)for i in range(depth)]
        self.fc_out =  nn.Linear(nHidden*2, nOut)
        self.module = nn.Sequential(self.fc_in, *self.hidden_layers)

    def forward(self, x):
        # input_ = input_.permute(2, 1, 0).contiguous()
        # pdb.set_trace()
        output = self.module(x)
        output = self.fc_out(output[-1])
        return output

class RCNN_Text(nn.Module):
    
    def __init__(self, nIn, nHidden):
        super(RCNN_Text, self).__init__()
        
       
        C = 4
        Ci = 1
        Co = 100
        Ks = [3, 4, 5]
    
        self.fc_in = nn.Linear(nIn, 2*nHidden)
        self.hidden_layers = [SimpleLSTM(nHidden*2, nHidden)for i in range(2)]
        self.rnn = nn.Sequential(*self.hidden_layers)

        self.conv13 = nn.Conv2d(Ci, Co, (3, 2*nHidden))
        self.conv14 = nn.Conv2d(Ci, Co, (4, 2*nHidden))
        self.conv15 = nn.Conv2d(Ci, Co, (5, 2*nHidden))
        
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(len(Ks)*Co, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        T, B, D = x.size()
        x = x.view(-1, D)
        x = self.fc_in(x) # T*hidden
        x = x.view(T, B,-1) # T*B*hidden
        recurrent  = self.rnn(x) #T*B*hidden
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit
# model = EmoNet(50, 256, 4, 2).cuda()