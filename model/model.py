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

class EmoNet(nn.Module):
    def __init__(self, nIn, nHidden, nOut, depth):
        super(EmoNet, self).__init__()
        self.fc_in = SimpleLinear(nIn, nHidden*2)
        self.hidden_layers = [SimpleLSTM(nHidden*2, nHidden)for i in range(depth)]
        self.fc_out =  nn.Linear(nHidden*2, nOut)
        self.module = nn.Sequential(self.fc_in, *self.hidden_layers)

    def forward(self, x):
        # input_ = input_.permute(2, 1, 0).contiguous()
        output = self.module(x)
        output = self.fc_out(output[-1])
        return output

# model = EmoNet(50, 256, 4, 2).cuda()