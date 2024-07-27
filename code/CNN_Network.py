import torch
import torch.nn as nn
import torch.nn.functional as F

#Muti-scale CNN
class CNN(nn.Module):
    def __init__(self, input_c,CNN_channel,num_layers, CNN_kernel_size):
        super(CNN, self).__init__()
        CNN_padding=[]
        for i in CNN_kernel_size:
            CNN_padding.append(int((i - 1) / 2))
        self.CNN_output_channel=CNN_channel[-1]
        self.net1,self.net2,self.net3= nn.ModuleList(),nn.ModuleList(),nn.ModuleList()
        for j in range(num_layers):
            if j==0:input_channel=input_c
            else:input_channel=CNN_channel[j-1]
            self.net1.append(nn.Conv1d(input_channel, CNN_channel[j], CNN_kernel_size[0], 1, CNN_padding[0]))
            self.net1.append(nn.BatchNorm1d(CNN_channel[j]))

            self.net2.append(nn.Conv1d(input_channel, CNN_channel[j], CNN_kernel_size[1], 1, CNN_padding[1]))
            self.net2.append(nn.BatchNorm1d(CNN_channel[j]))

            self.net3.append(nn.Conv1d(input_channel, CNN_channel[j], CNN_kernel_size[2], 1, CNN_padding[2]))
            self.net3.append(nn.BatchNorm1d(CNN_channel[j]))
            
        self.LeakyReLU = nn.LeakyReLU()
        
        self.res=nn.Sequential(nn.Conv1d(input_c, CNN_channel[-1], kernel_size=1, stride=1),nn.BatchNorm1d(CNN_channel[-1]))
    def forward(self, inputs_hybrid,inputs_esm2):
        scale1_result,scale2_result,scale3_result=inputs_hybrid,inputs_hybrid,inputs_hybrid
        esm_result=inputs_esm2
        for i in range(len(self.net1)):
            scale1_result=self.net1[i](scale1_result)#(B,c,l)
            scale2_result=self.net2[i](scale2_result)#(B,c,l)
            scale3_result=self.net3[i](scale3_result)#(B,c,l)
        if inputs_hybrid.shape[1]!=self.CNN_output_channel:
            scale1_result=self.res(inputs_hybrid)+scale1_result
            scale2_result=self.res(inputs_hybrid)+scale2_result
            scale3_result=self.res(inputs_hybrid)+scale3_result
        else:
            scale1_result=inputs_hybrid+scale1_result
            scale2_result=inputs_hybrid+scale2_result
            scale3_result=inputs_hybrid+scale3_result
        scale1_result=self.LeakyReLU(scale1_result)
        scale2_result=self.LeakyReLU(scale2_result)
        scale3_result=self.LeakyReLU(scale3_result)
        muti_result=torch.cat([scale1_result,scale2_result,scale3_result],1)#(B,3*C,w,l)
        return muti_result,esm_result

#Attention mechanism
class Attention(nn.Module):
    def __init__(self,in_planes,K):
        super(Attention,self).__init__()
        self.avgpool=nn.AdaptiveAvgPool1d(1)
        self.net=nn.Conv1d(in_planes,K,kernel_size=1)
        self.Softmax = nn.Softmax(dim=1)

    def forward(self,x):
        att=self.avgpool(x) #(bs,c,1)
        att=self.net(att).view(x.shape[0],-1)#(bs,K)
        return self.Softmax(att)
#Residual DYConv1D block
class Res_DYConv1d_block(nn.Module):
    def __init__(self,in_planes,out_planes,kernel_size,stride,dilation,K):
        super(Res_DYConv1d_block,self).__init__()
        self.in_planes=in_planes
        self.out_planes=out_planes
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=int((kernel_size - 1) / 2)
        self.dilation=dilation
        self.K=K#Experts num
        self.attention=Attention(in_planes=in_planes,K=K)

        self.weight=nn.Parameter(torch.randn(K,out_planes,in_planes,kernel_size),requires_grad=True)#(K,out_planes,in_planes,kernel_size)
        self.bias=nn.Parameter(torch.randn(K,out_planes),requires_grad=True)#(K,out_planes)
        self.res=nn.Sequential(nn.Conv1d(self.in_planes, self.out_planes, kernel_size=1, stride=1),nn.BatchNorm1d(self.out_planes))
        self.BN=nn.BatchNorm1d(self.out_planes)
        self.LeakyReLU = nn.LeakyReLU()
    def forward(self,x):
        input=x
        bs,in_c,l=x.shape#(B,c,l)
        softmax_att=self.attention(x)#(bs,K)
        x=x.view(1,-1,l)#(1,B*c,l)
        weight=self.weight.view(self.K,-1) #(K,out_planes*in_planes*kernel_size)
        aggregate_weight=torch.mm(softmax_att,weight).view(bs*self.out_planes,self.in_planes,self.kernel_size)#(bs*out_p,in_p,k)
        bias=self.bias.view(self.K,-1) #(K,out_p)
        aggregate_bias=torch.mm(softmax_att,bias).view(-1)#(bs*out_p)
        output=F.conv1d(x,weight=aggregate_weight,bias=aggregate_bias,stride=self.stride,padding=self.padding,groups=bs,dilation=self.dilation)
        output=self.BN(output.view(bs,self.out_planes,l))
        if in_c!=self.out_planes:
            output=self.LeakyReLU(output+self.res(input))
        else:output=self.LeakyReLU(output+input)
        return output
        
#DY-CNN
class DYConv1d(nn.Module):
    def __init__(self,in_planes,out_planes,kernel_size,dilation,K):
        super(DYConv1d,self).__init__()
        self.net= nn.ModuleList()
        for j in range(len(out_planes)):
            if j==0:input_channel=in_planes
            else:input_channel=out_planes[j-1]
            self.net.append(Res_DYConv1d_block(input_channel,out_planes[j],kernel_size,1,dilation,K))
    def forward(self, seq_input):
        result=seq_input
        for i in range(len(self.net)):
            result=self.net[i](result)#(B,c,l)
        return result 
