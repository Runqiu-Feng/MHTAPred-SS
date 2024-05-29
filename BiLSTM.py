import torch
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#BiLSTM model
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,dropout_rate):
        super(BiLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2
        self.batch_size = 1
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True,dropout=dropout_rate)
    def forward(self, input_seq,test):
        batch_size=len(input_seq)
        if test=='test':
            h_0 = torch.zeros(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
            c_0 = torch.zeros(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        else:
            h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
            c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        output, _ = self.lstm(input_seq, (h_0, c_0))
        return output
#BiGRU model
class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,dropout_rate):
        super(BiGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2
        self.batch_size = 1
        self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True,dropout=dropout_rate)
    def forward(self, input_seq,test):
        batch_size=len(input_seq)
        if test=='test':h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device)
        else:h0 = torch.randn(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device)
        output, _ = self.gru(input_seq, h0)
        return output
