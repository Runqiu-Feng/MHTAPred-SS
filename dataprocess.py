import h5py
import numpy as np
import torch
from tqdm import tqdm
import os
# phys_dict = {'A': [-0.350, -0.680, -0.677, -0.171, -0.170, 0.900, -0.476],
#                 'C': [-0.140, -0.329, -0.359, 0.508, -0.114, -0.652, 0.476],
#                 'D': [-0.213, -0.417, -0.281, -0.767, -0.900, -0.155, -0.635],
#                 'E': [-0.230, -0.241, -0.058, -0.696, -0.868, 0.900, -0.582],
#                 'F': [0.363, 0.373, 0.412, 0.646, -0.272, 0.155, 0.318],
#                 'G': [-0.900, -0.900, -0.900, -0.342, -0.179, -0.900, -0.900],
#                 'H': [0.384, 0.110, 0.138, -0.271, 0.195, -0.031, -0.106],
#                 'I': [0.900, -0.066, -0.009, 0.652, -0.186, 0.155, 0.688],
#                 'K': [-0.088, 0.066, 0.163, -0.889, 0.727, 0.279, -0.265],
#                 'L': [0.213, -0.066, -0.009, 0.596, -0.186, 0.714, -0.053],
#                 'M': [0.110, 0.066, 0.087, 0.337, -0.262, 0.652, -0.001],
#                 'N': [-0.213, -0.329, -0.243, -0.674, -0.075, -0.403, -0.529],
#                 'P': [0.247, -0.900, -0.294, 0.055, -0.010, -0.900, 0.106],
#                 'Q': [-0.230, -0.110, -0.020, -0.464, -0.276, 0.528, -0.371],
#                 'R': [0.105, 0.373, 0.466, -0.900, 0.900, 0.528, -0.371],
#                 'S': [-0.337, -0.637, -0.544, -0.364, -0.265, -0.466, -0.212],
#                 'T': [0.402, -0.417, -0.321, -0.199, -0.288, -0.403, 0.212],
#                 'V': [0.677, -0.285, -0.232, 0.331, -0.191, -0.031, 0.900],
#                 'W': [0.479, 0.900, 0.900, 0.900, -0.209, 0.279, 0.529],
#                 'Y':[0.363, 0.417, 0.541, 0.188, -0.274, -0.155, 0.476],
#                 'X': [0.0771, -0.1536, -0.0620, -0.0762, -0.1451, 0.0497, -0.0398]}

# for i in tqdm(range(12510)):
#     lihua_l=[]
#     psid='PS'+str(i)
#     for j in str(f_PISCES[psid][()])[2:-1]:
#         if j in list('ACDEFGHIKLMNPQRSTVWY'):
#             lihua_l.append(phys_dict[j])
#         else:
#             lihua_l.append(phys_dict['X'])
#     lihua_l=torch.from_numpy(np.array(lihua_l))
#     f_PISCES.create_dataset('lihua_new'+str(i), data=lihua_l)

# print(len(f_PISCES['PS100'][()]))

# n_SPOT_1D_Train=10029
# n_SPOT_1D_Valid=983
# n_Test2016=1213
# n_Test2018=250
# # 生成ESM-2嵌入向量数据，并将其写入文件中
# model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
# batch_converter = alphabet.get_batch_converter()
# model.eval()
# with h5py.File(r"/home/frq/paper1/datasets/SPOT_1D_Train.h5", "a") as f:
#     for i in tqdm(range(n_SPOT_1D_Train)):
#         id=str(i)##LETSHBG
#         seq=f['PS'+id][()]
#         seq=str(seq)[2:-1]
#         # batch_strs为原始序列字符串列表
#         # batch_tokens:对batch_strs进行编码的列表，0表示序列开头，2表示序列结尾，用1填充不够长的部分（长度由数据集中最长的序列决定）
#         data = [('0', seq)]
#         batch_labels, batch_strs, batch_tokens = batch_converter(data)
#         with torch.no_grad():
#             results = model(batch_tokens[0].unsqueeze(0), repr_layers=[33], return_contacts=True)
#         # token_representations是对batch_tokens[i]进行嵌入的结果
#         token_representations = results["representations"][33]
#         # # 将序列开头、结尾以及填充的嵌入向量删除
#         ESM_embedding=token_representations[0][1:len(seq) + 1].numpy()
#         f.create_dataset('ESM2_650M'+id, data=ESM_embedding)
# with h5py.File(r"/home/frq/paper1/datasets/SPOT_1D_Valid.h5", "a") as f:
#     for i in tqdm(range(n_SPOT_1D_Valid)):
#         id=str(i)##LETSHBG
#         seq=f['PS'+id][()]
#         seq=str(seq)[2:-1]
#         # batch_strs为原始序列字符串列表
#         # batch_tokens:对batch_strs进行编码的列表，0表示序列开头，2表示序列结尾，用1填充不够长的部分（长度由数据集中最长的序列决定）
#         data = [('0', seq)]
#         batch_labels, batch_strs, batch_tokens = batch_converter(data)
#         with torch.no_grad():
#             results = model(batch_tokens[0].unsqueeze(0), repr_layers=[33], return_contacts=True)
#         # token_representations是对batch_tokens[i]进行嵌入的结果
#         token_representations = results["representations"][33]
#         # # 将序列开头、结尾以及填充的嵌入向量删除
#         ESM_embedding=token_representations[0][1:len(seq) + 1].numpy()
#         f.create_dataset('ESM2_650M'+id, data=ESM_embedding)
# with h5py.File(r"/home/frq/paper1/datasets/Test2016.h5", "a") as f:
#     for i in tqdm(range(n_Test2016)):
#         id=str(i)##LETSHBG
#         seq=f['PS'+id][()]
#         seq=str(seq)[2:-1]
#         # batch_strs为原始序列字符串列表
#         # batch_tokens:对batch_strs进行编码的列表，0表示序列开头，2表示序列结尾，用1填充不够长的部分（长度由数据集中最长的序列决定）
#         data = [('0', seq)]
#         batch_labels, batch_strs, batch_tokens = batch_converter(data)
#         with torch.no_grad():
#             results = model(batch_tokens[0].unsqueeze(0), repr_layers=[33], return_contacts=True)
#         # token_representations是对batch_tokens[i]进行嵌入的结果
#         token_representations = results["representations"][33]
#         # # 将序列开头、结尾以及填充的嵌入向量删除
#         ESM_embedding=token_representations[0][1:len(seq) + 1].numpy()
#         f.create_dataset('ESM2_650M'+id, data=ESM_embedding)
# with h5py.File(r"/home/frq/paper1/datasets/Test2018.h5", "a") as f:
#     for i in tqdm(range(n_Test2018)):
#         id=str(i)##LETSHBG
#         seq=f['PS'+id][()]
#         seq=str(seq)[2:-1]
#         # batch_strs为原始序列字符串列表
#         # batch_tokens:对batch_strs进行编码的列表，0表示序列开头，2表示序列结尾，用1填充不够长的部分（长度由数据集中最长的序列决定）
#         data = [('0', seq)]
#         batch_labels, batch_strs, batch_tokens = batch_converter(data)
#         with torch.no_grad():
#             results = model(batch_tokens[0].unsqueeze(0), repr_layers=[33], return_contacts=True)
#         # token_representations是对batch_tokens[i]进行嵌入的结果
#         token_representations = results["representations"][33]
#         # # 将序列开头、结尾以及填充的嵌入向量删除
#         ESM_embedding=token_representations[0][1:len(seq) + 1].numpy()
#         f.create_dataset('ESM2_650M'+id, data=ESM_embedding)

# # 生成ASAquick预测结果
# with h5py.File(r"/home/frq/paper1/datasets/Test2018.h5", "r") as f1:
#     for i in tqdm(range(n_Test2018)):
#         id=str(i)
#         seq=f1['PS'+id][()]
#         # print(str(seq)[2:-1])
#         f = open(r'/home/frq/paper1/Fasta/Test2018/ID'+id+'.fasta', 'w')
#         f.write('>'+id+'\n')
#         f.write(str(seq)[2:-1])
#         f.close()
#         os.system('cd /home/frq/paper1/Fasta/Test2018'+';ASAquick '+'/home/frq/paper1/Fasta/Test2018/ID'+id+'.fasta')
        
# #将RASA预测结果写入.h5数据中
# #查看有没有空的预测结果,并将RASA写入.h5文件中
# with h5py.File(r"/home/frq/paper1/datasets/SPOT_1D_Valid.h5", "a") as f1:
#     for i in tqdm(range(n_SPOT_1D_Valid)):
#         id=str(i)
#         seq=f1['PS'+id][()]
#         file = open(r'/home/frq/paper1/Fasta/SPOT_1D_Valid/asaq.ID'+id+'.fasta'+'/rasaq.pred','r')#打开文件
#         file_data = file.readlines() #读取所有行
#         RASA=[]
#         for j in file_data:
#             x=j.split(' ')
#             if float(x[2])/2+0.5<=0.15:
#                 RASA.append(0)
#             else:RASA.append(1)
#         if len(seq)!=len(RASA):#查看有没有空的预测结果
#             print(id,i)
#         # f1.create_dataset('RASA'+id, data=RASA)