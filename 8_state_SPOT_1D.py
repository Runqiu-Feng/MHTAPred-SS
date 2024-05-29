import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import h5py
from random import shuffle
import MTL_STA
import warnings
import os
from early_stopping import EarlyStopping
torch.manual_seed(42)
torch.cuda.manual_seed(42)
warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#Read data
def read_datasets(dataset,ls,feature):
    data_X0=[]#one_hot
    data_X1=[]#hybrid
    data_X2=[]#esm-2
    data_Y1=[]#secondary structure
    data_Y2=[]#RSA
    protein_onehot = {'A': [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                'C': [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                'D': [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                'E': [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                'F': [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                'G': [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                'H': [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                'I': [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                'K': [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                'L': [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                'M': [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
                'N': [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
                'P': [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                'Q': [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
                'R': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
                'S': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
                'T': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
                'V': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
                'W': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
                'Y': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
                'X': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                'U': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                'B': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                'O': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]}
    dataset=eval(dataset)
    for i in tqdm(ls):
        data_seq_input=[]
        data_ss=[]
        data_pssm=dataset['Profile'+str(i)][:,:20].T
        data_hmm=dataset['Profile'+str(i)][:,20:50].T
        data_lihua=dataset['Profile'+str(i)][:,50:57].T
        if 'lihua' not in feature:data_inputs1=torch.from_numpy(np.concatenate((data_pssm, data_hmm), axis=0))
        elif 'hmm' not in feature:data_inputs1=torch.from_numpy(np.concatenate((data_pssm, data_lihua), axis=0))
        elif 'pssm' not in feature:data_inputs1=torch.from_numpy(np.concatenate((data_hmm, data_lihua), axis=0))
        else:data_inputs1=torch.from_numpy(np.concatenate((data_pssm, data_hmm, data_lihua), axis=0))
        data_esm2=torch.tensor(dataset['ESM2_650M'+str(i)][:].T)
        data_rasa=torch.tensor(dataset['RASA'+str(i)][:])
        for seq in str(dataset['PS'+str(i)][()])[2:-1]:
            data_seq_input.append(protein_onehot[seq])
        data_seq_input=torch.tensor(data_seq_input).t()
        for ss in str(dataset['SS'+str(i)][()])[2:-1]:
            data_ss.append(ss_dic[ss])
        data_ss=torch.tensor(data_ss)
        data_X0.append(data_seq_input)
        data_X1.append(data_inputs1)
        data_X2.append(data_esm2)
        data_Y1.append(data_ss)
        data_Y2.append(data_rasa)
    return data_X0,data_X1,data_X2,data_Y1,data_Y2
#Shuffle the data
def shuffle_data(V,W,X,Y,Z):
    c = list(zip(V,W,X,Y,Z))
    shuffle(c)
    V,W,X,Y,Z = zip(*c)
    return list(V),list(W),list(X), list(Y),list(Z)
#Divide minibatch
def split_list_average_n(inputs0_list,inputs1_list,inputs2_list,outputs1_list,outputs2_list, n):
    new_list0,new_list1,new_list2,new_list3,new_list4,mask_list = [],[],[],[],[],[]
    count=0
    for i in range(0, len(inputs1_list), n):
        l0,l1,l2,l3,l4=[],[],[],[],[]
        seq_len=[]
        for j in range(len(inputs1_list[i:i + n])):
            seq_len.append(inputs1_list[i:i + n][j].shape[1])
        max_len=max(seq_len)
        mask_list.append(torch.ones((len(inputs1_list[i:i + n]),max_len)))
        for k in range(len(inputs1_list[i:i + n])):
            length=inputs1_list[i:i + n][k].shape[1]
            if max_len!=length:
                len_dis=max_len-length
                vec_dis0=torch.zeros(inputs0_list[i:i + n][k].shape[0],len_dis)
                l0.append(torch.cat([inputs0_list[i:i + n][k],vec_dis0],1))
                vec_dis1=torch.zeros(inputs1_list[i:i + n][k].shape[0],len_dis)
                l1.append(torch.cat([inputs1_list[i:i + n][k],vec_dis1],1))
                vec_dis2=torch.zeros(inputs2_list[i:i + n][k].shape[0],len_dis)
                l2.append(torch.cat([inputs2_list[i:i + n][k],vec_dis2],1))
                vec_dis3,vec_dis4=torch.zeros(len_dis),torch.zeros(len_dis)
                l3.append(torch.cat([outputs1_list[i:i + n][k],vec_dis3],0))
                l4.append(torch.cat([outputs2_list[i:i + n][k],vec_dis4],0))
                mask_list[count][k][length:]=0
            else:
                l0.append(inputs0_list[i:i + n][k])
                l1.append(inputs1_list[i:i + n][k])
                l2.append(inputs2_list[i:i + n][k])
                l3.append(outputs1_list[i:i + n][k])
                l4.append(outputs2_list[i:i + n][k])
        new_list0.append(torch.stack(l0,0))
        new_list1.append(torch.stack(l1,0))
        new_list2.append(torch.stack(l2,0))
        new_list3.append(torch.stack(l3,0))
        new_list4.append(torch.stack(l4,0))
        count=count+1
    return new_list0,new_list1,new_list2,new_list3,new_list4,mask_list
#Weight initialization
def weights_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data)
#STA model loading
def load_model_pre(save_name, model):
    model_data = torch.load(save_name)
    NL=[]
    for i in model_data['model_dict'].keys():
        if 'CNN' not in i:
            NL.append(i)
    for j in NL:
        model_data['model_dict'].pop(j)
    model.load_state_dict(model_data['model_dict'])
    print("model load success")
#PSSP-MTL model loading
def load_model(save_name, model):
    model_data = torch.load(save_name)
    model.load_state_dict(model_data['model_dict'])
    print("model load success")
#Model testing
def test(test_X0,test_X1,test_X2,test_Y1,test_Y2,model_PSSP,loss_function):
    model_PSSP.eval()
    with torch.no_grad():
        sum_true=0
        sum_res=0
        G_true=0
        G_res=0
        H_true=0
        H_res=0
        I_true=0
        I_res=0
        T_true=0
        T_res=0
        E_true=0
        E_res=0
        B_true=0
        B_res=0
        S_true=0
        S_res=0
        L_true=0
        L_res=0
        loss=0
        loss1=0
        loss2=0
        ss2num_dic={0:'G',1:'H',2:'I',3:'T',4:'E',5:'B',6:'S',7:'L',8:'X'}
        with open('sov_Q8.txt', 'w') as f:
            no_rasa_num=0
            for i in tqdm(range(len(test_X1))):
                inputs_onehot=test_X0[i][None].to(torch.float).to(device)
                inputs_hybrid=test_X1[i][None].to(torch.float).to(device)
                inputs_esm2=test_X2[i][None].to(torch.float).to(device)
                out1, out2=model_PSSP(inputs_onehot,inputs_hybrid,inputs_esm2,'test')#(B,8,l)/(B,2,l)
                label1=test_Y1[i][None].to(torch.long).to(device)#(B,l)
                label1_mask=torch.zeros((label1.shape[0],label1.shape[1])).to(torch.long).to(device)
                label1_loss=torch.where(label1 == 8, label1_mask, label1)
                label2=test_Y2[i][None].to(torch.long).to(device)#(B,l)
                if label2.shape[1]==0:
                    l2=0
                    no_rasa_num=no_rasa_num+1
                else:l2=loss_function(out2,label2).sum()/len(label2[0])
                loss1=loss1+loss_function(out1,label1_loss).sum()/len(label1_loss[0])
                loss2=loss2+l2
                loss=loss+(0.8*loss_function(out1,label1_loss).sum()/len(label1_loss[0]))+(0.2*l2)
                pre1=torch.argmax(out1[0],dim=0)
                label_t=''
                label_p=''
                for j in range(len(pre1)):
                    label_t=label_t+ss2num_dic[int(label1[0][j].item())]
                    label_p=label_p+ss2num_dic[int(pre1[j])]
                    if pre1[j]==label1[0][j]:
                        sum_true=sum_true+1
                        if pre1[j]==0:
                            G_true=G_true+1
                        elif pre1[j]==1:
                            H_true=H_true+1
                        elif pre1[j]==2:
                            I_true=I_true+1
                        elif pre1[j]==3:
                            T_true=T_true+1
                        elif pre1[j]==4:
                            E_true=E_true+1
                        elif pre1[j]==5:
                            B_true=B_true+1
                        elif pre1[j]==6:
                            S_true=S_true+1
                        elif pre1[j]==7:
                            L_true=L_true+1
                    if label1[0][j]==0:
                        G_res=G_res+1
                        sum_res=sum_res+1
                    if label1[0][j]==1:
                        H_res=H_res+1
                        sum_res=sum_res+1
                    if label1[0][j]==2:
                        I_res=I_res+1
                        sum_res=sum_res+1
                    if label1[0][j]==3:
                        T_res=T_res+1
                        sum_res=sum_res+1
                    if label1[0][j]==4:
                        E_res=E_res+1
                        sum_res=sum_res+1
                    if label1[0][j]==5:
                        B_res=B_res+1
                        sum_res=sum_res+1
                    if label1[0][j]==6:
                        S_res=S_res+1
                        sum_res=sum_res+1
                    if label1[0][j]==7:
                        L_res=L_res+1
                        sum_res=sum_res+1
                f.write('>'+str(i)+' '+str(len(pre1))+'\n')#先target后predicate
                f.write(label_t+'\n')
                f.write(label_p+'\n')
        os.system('perl SOV.pl sov_Q8.txt')
        Q8_score=100*sum_true/(sum_res+1e-12)
        QG_score=100*G_true/(G_res+1e-12)
        QH_score=100*H_true/(H_res+1e-12)
        QI_score=100*I_true/(I_res+1e-12)
        QT_score=100*T_true/(T_res+1e-12)
        QE_score=100*E_true/(E_res+1e-12)
        QB_score=100*B_true/(B_res+1e-12)
        QS_score=100*S_true/(S_res+1e-12)
        QL_score=100*L_true/(L_res+1e-12)
        loss=loss/len(test_X1)
        loss1=loss1/len(test_X1)
        loss2=loss2/(len(test_X1)-no_rasa_num)
        f_sov = open("sov_Q8_Eval.txt")
        line = f_sov.readline()
        sov_score=line.strip()
    return Q8_score, QG_score, QH_score, QI_score, QT_score, QE_score, QB_score, QS_score,QL_score,loss,loss1,loss2,sov_score
#Model training
def train(epoch1,epochs,train_X0,train_X1,train_X2,train_Y1,train_Y2,valid_X0,valid_X1,valid_X2,valid_Y1,valid_Y2,model_PSSP,loss_function,optim,scheduler_model,early):
    global batchsize
    for epoch in range(epoch1,epochs):
        loss_epoch,loss1_epoch,loss2_epoch=0,0,0
        model_PSSP.train()
        X0_shuffle,X1_shuffle,X2_shuffle,Y1_shuffle,Y2_shuffle=shuffle_data(train_X0,train_X1,train_X2,train_Y1,train_Y2)
        X0_shuffle,X1_shuffle,X2_shuffle,Y1_shuffle,Y2_shuffle,mask_shuffle=split_list_average_n(X0_shuffle,X1_shuffle,X2_shuffle,Y1_shuffle,Y2_shuffle, batchsize)
        for i in tqdm(range(len(X1_shuffle))):
            inputs_onehot=X0_shuffle[i].to(torch.float).to(device)
            inputs_hybrid=X1_shuffle[i].to(torch.float).to(device)
            inputs_esm2=X2_shuffle[i].to(torch.float).to(device)
            out1, out2=model_PSSP(inputs_onehot,inputs_hybrid,inputs_esm2,0)#(B,3,l)/(B,2,l)
            label1=Y1_shuffle[i].to(torch.long).to(device)#(B,l)
            label2=Y2_shuffle[i].to(torch.long).to(device)#(B,l)
            mask=mask_shuffle[i].to(torch.long).to(device)
            num=len(torch.nonzero(mask))
            loss1=torch.mul(loss_function(out1,label1),mask).sum()/num
            loss2=torch.mul(loss_function(out2,label2),mask).sum()/num
            loss=0.8*loss1+0.2*loss2
            loss_epoch=loss_epoch+loss.item()
            loss1_epoch=loss1_epoch+loss1.item()
            loss2_epoch=loss2_epoch+loss2.item()
            optim.zero_grad()
            loss.backward()
            optim.step()
        loss_epoch=loss_epoch/len(X1_shuffle)
        loss1_epoch=loss1_epoch/len(X1_shuffle)
        loss2_epoch=loss2_epoch/len(X1_shuffle)
        _, _, _, _, _, _, _, _,_,loss_valid,_,_,_=test(valid_X0,valid_X1,valid_X2,valid_Y1,valid_Y2,model_PSSP,loss_function)
        if os.path.exists('sov_Q8_Eval.txt'):
            os.remove('sov_Q8_Eval.txt')
        # print('epoch:', epoch,'Training is complete, start testing——————————————————————————————————————————————————————————————————————')
        # test_result=test(test_X0,test_X1,test_X2,test_Y1,test_Y2,model_PSSP,loss_function)
        # if os.path.exists('sov_Q8_Eval.txt'):
        #    os.remove('sov_Q8_Eval.txt')
        scheduler_model.step()
        torch.cuda.empty_cache()
        early(loss_valid.item())
        if early.early_stop:
            print("Early stopping")
            break

#读取数据
print('Reading training & validation datasets')
SPOT_1D_Train = h5py.File(r"../datasets/SPOT_1D_Train.h5", "r")
SPOT_1D_Valid = h5py.File(r"../datasets/SPOT_1D_Valid.h5", "r")
Test2016 = h5py.File(r"../datasets/Test2016.h5", "r")
Test2018 = h5py.File(r"../datasets/Test2018.h5", "r")
ss_dic={'G':0,'H':1,'I':2,'T':3,'E':4,'B':5,'S':6,'L':7,'X':8}
#每条数据集中包含的序列数目
batchsize=32#批量大小
n_SPOT_1D_Train=10029
n_SPOT_1D_Valid=983
n_Test2016=1213
n_Test2018=250
SPOT_1D_Train_ls=range(n_SPOT_1D_Train)
SPOT_1D_Valid_ls=range(n_SPOT_1D_Valid)
Test2016_ls=range(n_Test2016)
Test2018_ls=range(n_Test2018)
SPOT_1D_valid_X0,SPOT_1D_valid_X1,SPOT_1D_valid_X2,SPOT_1D_valid_Y1,SPOT_1D_valid_Y2=read_datasets('SPOT_1D_Valid',SPOT_1D_Valid_ls)
SPOT_1D_train_X0,SPOT_1D_train_X1,SPOT_1D_train_X2,SPOT_1D_train_Y1,SPOT_1D_train_Y2=read_datasets('SPOT_1D_Train',SPOT_1D_Train_ls)
Test2016_test_X0,Test2016_test_X1,Test2016_test_X2,Test2016_test_Y1,Test2016_test_Y2=read_datasets('Test2016',Test2016_ls)
Test2018_test_X0,Test2018_test_X1,Test2018_test_X2,Test2018_test_Y1,Test2018_test_Y2=read_datasets('Test2018',Test2018_ls)
DY_CNN_in_planes=21
DY_CNN_out_planes=[8,4]
DY_CNN_kernel_size=3
DY_CNN_dilation=1
DY_CNN_K=3
BLSTM_input_size=DY_CNN_out_planes[-1]
BLSTM_hidden_size=20
BLSTM_num_layers=2
BLSTM_dropout_rate=0.5
fc_dim=100
Encoder = MTL_STA.Encoder(DY_CNN_in_planes,DY_CNN_out_planes,DY_CNN_kernel_size,DY_CNN_dilation,DY_CNN_K,BLSTM_input_size, BLSTM_hidden_size, BLSTM_num_layers,BLSTM_dropout_rate,fc_dim)
Encoder.to(device)
load_model_pre('../model/Encoder/SPOT_1D_Encoder.pth', Encoder)
for name, value in Encoder.named_parameters():
    value.requires_grad = False
#hybrid=>1DCNN
CNN_channel=[90,110,150,200]
CNN_kernel_size=[1,5,9]
#hybrid+esm2=>MMoE
input_c=57+DY_CNN_out_planes[-1]
mmoe_input_channel=CNN_channel[-1]*len(CNN_kernel_size)+1280
mmoe_expert_num=3
mmoe_channel=[1024,512]
mmoe_layers=2
mmoe_kernel_size=3
#MMoE=>TCN-BiLSTM
#TCN-BiLSTM(PSSP)
TCN_num_inputs1=mmoe_channel[-1]
TCN_num_channels1=[256,256]
TCN_kernel_size1=5
TCN_dropout1=0.3
gru_input_size1=TCN_num_channels1[-1]
gru_hidden_size1=180
gru_num_layers1=5
#TCN-BiLSTM(RASA)
TCN_num_inputs2=mmoe_channel[-1]
TCN_num_channels2=[32,32]
TCN_kernel_size2=5
TCN_dropout2=0.5
gru_input_size2=TCN_num_channels2[-1]
gru_hidden_size2=16
gru_num_layers2=3
fc1_dim=180
fc2_dim=8
LR=0.0005
gru_dropout1=0.3
gru_dropout2=0.8
fc_dropout1=0.3
fc_dropout2=0.8
PSSP = MTL_STA.PSSP_Q8(input_c, CNN_channel,len(CNN_channel), CNN_kernel_size,mmoe_input_channel,mmoe_expert_num, mmoe_channel,mmoe_layers,mmoe_kernel_size,
                TCN_num_inputs1, TCN_num_channels1, TCN_kernel_size1, TCN_dropout1,
                TCN_num_inputs2, TCN_num_channels2, TCN_kernel_size2, TCN_dropout2,
                gru_input_size1, gru_hidden_size1, gru_num_layers1,gru_dropout1,
                gru_input_size2, gru_hidden_size2, gru_num_layers2,gru_dropout2,fc_dropout1,fc_dropout2,fc1_dim,fc2_dim,Encoder)
weights_init(PSSP)
PSSP.to(device)
entropy_loss = nn.CrossEntropyLoss(reduction='none')
opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, PSSP.parameters()), lr=LR, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)
early_stopping = EarlyStopping(patience=3)
print('Model training starts:')
train(0,1000,SPOT_1D_train_X0,SPOT_1D_train_X1,SPOT_1D_train_X2,SPOT_1D_train_Y1,SPOT_1D_train_Y2,SPOT_1D_valid_X0,SPOT_1D_valid_X1,SPOT_1D_valid_X2,SPOT_1D_valid_Y1,SPOT_1D_valid_Y2,PSSP,entropy_loss,opt,scheduler,early_stopping)