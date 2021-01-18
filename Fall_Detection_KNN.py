import pandas as pd
import numpy as np
import sklearn
#import glob
#import os
import matplotlib.pyplot as plt
from sklearn import datasets, neighbors
from mlxtend.plotting import plot_decision_regions

#Step 1 Preprocessing of the Data

path1 = "/content/drive/MyDrive/SisFall/nf_combined.csv"
path2 = "/content/drive/MyDrive/SisFall/f_combined.csv"
nf_df = pd.read_csv(path1, header =0)
f_df = pd.read_csv(path2,header = 0)

#Converting 1st Column to float # Correcting Error in writing of csv file
nf_df["0"] = pd.to_numeric(nf_df["0"], errors='coerce')
#nf_df['0'] = nf_df['0'].astype(float)

#Remvoing NaN Values
nf_df = nf_df.dropna(axis =0) 
f_df = f_df.dropna(axis =0) 

#removing the data from sensor 2
nf_df = nf_df.drop(["6","7","8"], axis = 1) 
f_df = f_df.drop(["6","7","8"],axis =1) 


# Converting AD values
#Acceleration [g]: [(2*Range)/(2^Resolution)]*AD
#Angular velocity [°/s]: [(2*Range)/(2^Resolution)]*RD
t_acc= ((2*16)/2**13)   
r_acc= ((2*2000)/2**16)
nf_df.iloc[:,0:4] *= t_acc
nf_df.iloc[:,4:6] *= r_acc 
f_df.iloc[:,0:4] *= t_acc 
f_df.iloc[:,4:6] *= r_acc

#Normalization of the Data
nf_minm = nf_df.min(axis =0)
nf_maxm = nf_df.max(axis =0)
nf_df = (nf_df-nf_minm)/(nf_maxm-nf_minm)

f_minm = f_df.min(axis =0)
f_maxm = f_df.max(axis =0)
f_df = (f_df-f_minm)/(f_maxm-f_minm)

#For Sckit Learning
nf_target = pd.DataFrame(np.zeros((len(nf_df),1)))
f_target = pd.DataFrame(np.ones((len(f_df),1)))

#Splitting Data into Training and Testing
nf_rows = np.size(nf_df,axis =0)
f_rows = np.size(f_df,axis =0)

nf_df_train = nf_df.iloc[0:int(0.3*nf_rows),:]
f_df_train = f_df.iloc[0:int(0.3*f_rows),:]
df_train = pd.concat([nf_df_train, f_df_train], axis=0, sort=False)
nf_train_target = nf_target.iloc[0:int(0.3*nf_rows),:]
f_train_target = f_target.iloc[0:int(0.3*f_rows),:]
df_train_target = pd.concat([nf_train_target, f_train_target], axis=0, sort=False)

nf_df_test = nf_df.iloc[int(0.3*nf_rows):,:]
f_df_test = f_df.iloc[int(0.3*f_rows):,:]
df_test = pd.concat([nf_df_test, f_df_test], axis=0, sort=False)
nf_test_target = nf_target.iloc[int(0.3*nf_rows):,:]
f_test_target = f_target.iloc[int(0.3*f_rows):,:]
df_test_target = pd.concat([nf_test_target, f_test_target], axis=0, sort=False)

#KNN
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(df_train, df_train_target)

#Predictions

y_pred = classifier.predict(df_test)

#Accuracy Check
from sklearn.metrics import r2_score

r2_score(df_test_target, y_pred)
