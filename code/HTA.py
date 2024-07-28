import CNN_Network
import BiLSTM
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self,DY_CNN_in_planes,DY_CNN_out_planes,DY_CNN_kernel_size,DY_CNN_dilation,DY_CNN_K,BLSTM_input_size, BLSTM_hidden_size, BLSTM_num_layers,BLSTM_dropout_rate,fc_dim):
        super(Encoder, self).__init__()
        self.LeakyReLU=nn.LeakyReLU()
        self.CNN = CNN_Network.DYConv1d(DY_CNN_in_planes,DY_CNN_out_planes,DY_CNN_kernel_size,DY_CNN_dilation,DY_CNN_K)
        self.BLSTM=BiLSTM.BiLSTM(BLSTM_input_size, BLSTM_hidden_size, BLSTM_num_layers,BLSTM_dropout_rate)
        self.fc = nn.Sequential(nn.Linear(BLSTM_hidden_size*2, fc_dim),nn.LeakyReLU(),nn.Linear(fc_dim, 21),nn.LeakyReLU())
    def forward(self, input,test):
        output1 = self.CNN(input).transpose(1,2)#(B/l/c)DYConv1d
        output2 = self.BLSTM(output1,test)#(B/l/c)BiLSTM
        output = self.fc(output2).transpose(1,2)#(B/c/L)fc
        return output