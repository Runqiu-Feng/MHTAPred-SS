import CNN_Network
import MMoE
import TCN
import torch
import BiLSTM
import torch.nn as nn

#3_state_PSSP
class PSSP_Q3(nn.Module):
    def __init__(self,input_c, CNN_channel,num_layers, CNN_kernel_size,mmoe_input_channel,mmoe_expert_num, mmoe_channel,mmoe_layers,mmoe_kernel_size,
                 TCN_num_inputs1, TCN_num_channels1, TCN_kernel_size1, TCN_dropout1,
                 TCN_num_inputs2, TCN_num_channels2, TCN_kernel_size2, TCN_dropout2,
                 gru_input_size1, gru_hidden_size1, gru_num_layers1,gru_dropout1,
                 gru_input_size2, gru_hidden_size2, gru_num_layers2,gru_dropout2,fc_dropout1,fc_dropout2,fc1_dim,fc2_dim,encoder_model):
        super(PSSP_Q3, self).__init__()
        self.LeakyReLU=nn.LeakyReLU()
        self.dropout1=nn.Dropout1d(p=fc_dropout1)
        self.dropout2=nn.Dropout1d(p=fc_dropout2)
        self.CNN = CNN_Network.CNN(input_c,CNN_channel,num_layers, CNN_kernel_size)
        self.encoder=encoder_model
        self.MMoE=MMoE.MMoE(mmoe_input_channel,mmoe_expert_num, mmoe_channel,mmoe_layers,mmoe_kernel_size)
        self.TCN1=TCN.TemporalConvNet(TCN_num_inputs1, TCN_num_channels1, TCN_kernel_size1, TCN_dropout1)
        self.TCN2=TCN.TemporalConvNet(TCN_num_inputs2, TCN_num_channels2, TCN_kernel_size2, TCN_dropout2)
        self.BiGRU1=BiLSTM.BiGRU(gru_input_size1, gru_hidden_size1, gru_num_layers1,gru_dropout1)
        self.BiGRU2=BiLSTM.BiGRU(gru_input_size2, gru_hidden_size2, gru_num_layers2,gru_dropout2)
        self.fc1_1 = nn.Sequential(nn.Linear(gru_hidden_size1*2, fc1_dim),nn.LeakyReLU(),nn.Linear(fc1_dim, fc1_dim))
        self.fc1_2 = nn.Linear(fc1_dim, 3)
        self.fc2_1 = nn.Sequential(nn.Linear(gru_hidden_size2*2, fc2_dim),nn.LeakyReLU(),nn.Linear(fc2_dim, fc2_dim))
        self.fc2_2 = nn.Linear(fc2_dim, 2)
    def forward(self,inputs_onehot,inputs_hybrid,inputs_esm2,test):
        y_encoder=self.encoder(inputs_onehot,test)#(B,c,L)
        inputs=torch.cat([inputs_hybrid,y_encoder],1)#(B,c,L)
        y1,y2 = self.CNN(inputs,inputs_esm2)#(B,c,l)
        y3=torch.cat([y1,y2],1)#(B,c,l)
        out1,out2=self.MMoE(y3)#out1/out2:(B,c,l)
        out1=self.TCN1(out1).transpose(1,2)#out1:(B,l,c)
        out2=self.TCN2(out2).transpose(1,2)#out2:(B,l,c)
        out1=self.BiGRU1(out1,test)#out1:(B,l,c)
        out2=self.BiGRU2(out2,test)#out2:(B,l,c)
        out1=self.fc1_1(out1).transpose(1,2)#out1:(B,fc1_dim,l)
        out2=self.fc2_1(out2).transpose(1,2)#out2:(B,fc2_dim,l)
        out1=self.dropout1(out1).transpose(1,2)#out1:(B,l,fc1_dim)
        out2=self.dropout2(out2).transpose(1,2)#out2:(B,l,fc2_dim)
        out1=self.fc1_2(out1).transpose(1,2)#out1:(B,3,l)
        out2=self.fc2_2(out2).transpose(1,2)#out1:(B,2,l)
        out1=self.LeakyReLU(out1)
        out2=self.LeakyReLU(out2)
        return out1, out2

#8_state_PSSP  
class PSSP_Q8(nn.Module):
    def __init__(self,input_c, CNN_channel,num_layers, CNN_kernel_size,mmoe_input_channel,mmoe_expert_num, mmoe_channel,mmoe_layers,mmoe_kernel_size,
                 TCN_num_inputs1, TCN_num_channels1, TCN_kernel_size1, TCN_dropout1,
                 TCN_num_inputs2, TCN_num_channels2, TCN_kernel_size2, TCN_dropout2,
                 gru_input_size1, gru_hidden_size1, gru_num_layers1,gru_dropout1,
                 gru_input_size2, gru_hidden_size2, gru_num_layers2,gru_dropout2,fc_dropout1,fc_dropout2,fc1_dim,fc2_dim,encoder_model):
        super(PSSP_Q8, self).__init__()
        self.LeakyReLU=nn.LeakyReLU()
        self.dropout1=nn.Dropout1d(p=fc_dropout1)
        self.dropout2=nn.Dropout1d(p=fc_dropout2)
        self.CNN = CNN_Network.CNN(input_c,CNN_channel,num_layers, CNN_kernel_size)
        self.encoder=encoder_model
        self.MMoE=MMoE.MMoE(mmoe_input_channel,mmoe_expert_num, mmoe_channel,mmoe_layers,mmoe_kernel_size)
        self.TCN1=TCN.TemporalConvNet(TCN_num_inputs1, TCN_num_channels1, TCN_kernel_size1, TCN_dropout1)
        self.TCN2=TCN.TemporalConvNet(TCN_num_inputs2, TCN_num_channels2, TCN_kernel_size2, TCN_dropout2)
        self.BiGRU1=BiLSTM.BiGRU(gru_input_size1, gru_hidden_size1, gru_num_layers1,gru_dropout1)
        self.BiGRU2=BiLSTM.BiGRU(gru_input_size2, gru_hidden_size2, gru_num_layers2,gru_dropout2)
        self.fc1_1 = nn.Sequential(nn.Linear(gru_hidden_size1*2, fc1_dim),nn.LeakyReLU(),nn.Linear(fc1_dim, fc1_dim))
        self.fc1_2 = nn.Linear(fc1_dim, 8)
        self.fc2_1 = nn.Sequential(nn.Linear(gru_hidden_size2*2, fc2_dim),nn.LeakyReLU(),nn.Linear(fc2_dim, fc2_dim))
        self.fc2_2 = nn.Linear(fc2_dim, 2)
    def forward(self,inputs_onehot,inputs_hybrid,inputs_esm2,test):
        y_encoder=self.encoder(inputs_onehot,test)#(B,c,L)
        inputs=torch.cat([inputs_hybrid,y_encoder],1)#(B,c,L)
        y1,y2 = self.CNN(inputs,inputs_esm2)#(B,c,l)
        y3=torch.cat([y1,y2],1)#(B,c,l)
        out1,out2=self.MMoE(y3)#out1/out2:(B,c,l)
        out1=self.TCN1(out1).transpose(1,2)#out1:(B,l,c)
        out2=self.TCN2(out2).transpose(1,2)#out2:(B,l,c)
        out1=self.BiGRU1(out1,test)#out1:(B,l,c)
        out2=self.BiGRU2(out2,test)#out2:(B,l,c)
        out1=self.fc1_1(out1).transpose(1,2)#out1:(B,fc1_dim,l)
        out2=self.fc2_1(out2).transpose(1,2)#out2:(B,fc2_dim,l)
        out1=self.dropout1(out1).transpose(1,2)#out1:(B,l,fc1_dim)
        out2=self.dropout2(out2).transpose(1,2)#out2:(B,l,fc2_dim)
        out1=self.fc1_2(out1).transpose(1,2)#out1:(B,3,l)
        out2=self.fc2_2(out2).transpose(1,2)#out1:(B,2,l)
        out1=self.LeakyReLU(out1)
        out2=self.LeakyReLU(out2)
        return out1, out2

#HTA model
class Encoder(nn.Module):
    def __init__(self,DY_CNN_in_planes,DY_CNN_out_planes,DY_CNN_kernel_size,DY_CNN_dilation,DY_CNN_K,BLSTM_input_size, BLSTM_hidden_size, BLSTM_num_layers,BLSTM_dropout_rate,fc_dim):
        super(Encoder, self).__init__()
        self.LeakyReLU=nn.LeakyReLU()
        self.CNN = CNN_Network.DYConv1d(DY_CNN_in_planes,DY_CNN_out_planes,DY_CNN_kernel_size,DY_CNN_dilation,DY_CNN_K)
    def forward(self, input,test):
        output1 = self.CNN(input)#(B/c/l)
        return output1