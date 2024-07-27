# MHTAPred-SS
### 1. Datasets
Public datasets used in this paper:
|Types|PISCES|CASP12|CASP13|CASP14|CB513|SPOT_1D_Train|SPOT_1D_Valid|TEST2016|TEST2018|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|#Sequences|12510|47|41|33|513|10029|983|1213|250|
|#Residues|:-:|13718|12217|9049|84119|:-:|:-:|287877|56654|

### 2. Dependency
MTL-STA is developed under Ubuntu environment with:
```
python==3.7.13
torch==1.13.1
torchaudio==0.13.1
torchvision==0.14.1
biopython==1.81
h5py==3.8.0
numpy==1.21.6
pandas==1.3.5
tqdm==4.66.1
matplotlib==3.5.3
seaborn==0.12.2
```
### 3. Usage
#### Step 1
Download and unzip the dataset and source code.
#### Step 2
Install all dependencies that meet the required versions.
#### Step 3
Preprocess the raw data and organize them into .h5 files containing PSSM, HMM, physicochemical properties and embedded features. The specific commands are as follows:
```
python dataprocess.py
```
#### Step 4
Use the following commands to train and test our proposed protein secondary result prediction model:
```
python 3_state_PISCES.py
python 8_state_PISCES.py
python 3_state_SPOT_1D.py
python 8_state_SPOT_1D.py
```
### 4. References
[1] Wei Yang,Yang Liu, and Chunjing Xiao, Deep Metric Learning for Accurate Protein Secondary Structure Prediction,Knowledge-based systems,2022. 242: p. 108356.

[2] Hanson, J., et al., Improving prediction of protein secondary structure, backbone angles, solvent accessibility and contact numbers by using predicted contact maps and an ensemble of recurrent and residual convolutional neural networks. Bioinformatics, 2019. 35(14): p. 2403-2410.

[3] Liu, T. and Z. Wang, SOV_refine: A further refined definition of segment overlap score and its significance for protein structure similarity. Source Code for Biology and Medicine, 2018. 13(1).
