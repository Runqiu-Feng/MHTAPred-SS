import torch
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#MMoE_gate
def gate_output(experts, g1, g2):
    out1=torch.zeros((experts[0].shape[0],experts[0].shape[1],experts[0].shape[2])).to(device)#(B,c,l)
    out2=torch.zeros((experts[0].shape[0],experts[0].shape[1],experts[0].shape[2])).to(device)#(B,c,l)
    for i in range(len(g1)):
        x1=experts[0][0]*g1[0][0]
        x2=experts[0][0]*g2[0][0]
        for j in range(1,len(experts)):
            x1=x1+experts[j][i]*g1[i][j]#(c,l)
            x2=x2+experts[j][i]*g2[i][j]#(c,l)
        out1[i]=x1
        out2[i]=x2
    return out1, out2

#MMoE model
class MMoE(nn.Module):
    def __init__(self, mmoe_input_channel,mmoe_expert_num, mmoe_channel,mmoe_layers,mmoe_kernel_size):
        super(MMoE, self).__init__()
        self.LeakyReLU = nn.LeakyReLU()
        self.mmoe_channel=mmoe_channel[-1]
        self.netlist=nn.ModuleList()
        for k in range(mmoe_expert_num):
            self.netlist.append(nn.ModuleList())
        mmoe_padding=(mmoe_kernel_size-1)//2
        for i in range(mmoe_expert_num):
            for j in range(mmoe_layers):
                if j==0:input_channel=mmoe_input_channel
                else:input_channel=mmoe_channel[j-1]
                self.netlist[i].append(nn.Conv1d(input_channel, mmoe_channel[j], mmoe_kernel_size, 1, mmoe_padding))
                self.netlist[i].append(nn.BatchNorm1d(mmoe_channel[j]))
        self.gate1=nn.Sequential(nn.Conv1d(mmoe_input_channel, mmoe_expert_num, mmoe_kernel_size, 1, mmoe_padding),nn.BatchNorm1d(mmoe_expert_num), nn.LeakyReLU(), nn.Softmax(dim=1))
        self.gate2=nn.Sequential(nn.Conv1d(mmoe_input_channel, mmoe_expert_num, mmoe_kernel_size, 1, mmoe_padding),nn.BatchNorm1d(mmoe_expert_num), nn.LeakyReLU(), nn.Softmax(dim=1))
        self.res=nn.Sequential(nn.Conv1d(mmoe_input_channel, mmoe_channel[-1], kernel_size=1, stride=1),nn.BatchNorm1d(mmoe_channel[-1]))
    def forward(self, x):
        experts_result=[]
        for i in self.netlist:
            inputs=x
            for j in range(len(i)):
                inputs=i[j](inputs)
            experts_result.append(inputs)
        g1=self.gate1(x)#(B,expert_num,l)
        g2=self.gate2(x)#(B,expert_num,l)
        out1, out2=gate_output(experts_result,g1,g2)
        if x.shape[1]!=self.mmoe_channel:
            out1=self.LeakyReLU(out1+self.res(x))
            out2=self.LeakyReLU(out2+self.res(x))
        else:
            out1=self.LeakyReLU(out1+x)
            out2=self.LeakyReLU(out2+x)
        return out1, out2