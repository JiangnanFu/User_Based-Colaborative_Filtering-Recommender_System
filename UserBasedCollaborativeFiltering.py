#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 18:51:59 2018

@author: sanjeet
"""
import pandas as pd
import numpy as np
from math import sqrt
import time
from numpy import random
from sklearn.metrics import mean_absolute_error,mean_squared_error
np.set_printoptions(formatter={'float_kind':'{:f}'.format})
e=np.exp
rmed=3


#rating=np.array(([4,3,5,4],[5,3,0,0],[4,3,3,4],[2,1,0,0],[4,2,0,0]))
cols=['userId','itemId','rating','timestamp']
df=pd.read_table('ml-100k/u.data',sep='\t',header=None,names=cols)
df.drop(columns=['timestamp'],inplace=True)
n_users=df['userId'].unique().shape[0]
n_movie=df['itemId'].unique().shape[0]
print(n_users,n_movie)


rating=np.zeros((n_users,n_movie))

mat_start=time.time()
for  row in df.itertuples():
    rating[row[1]-1,row[2]-1]=row[3]
mat_end=time.time()
print("Matrix conversion time = ",mat_end-mat_start)
trainset = np.copy(rating)
testset=np.zeros((n_users,n_movie))

#z for iterate over each row
train_test=time.time()
z=0
for row in rating:
    nz_in=np.nonzero(row)
    per_20=int(len(nz_in[0])*0.2)
    rand =random.choice(nz_in[0],per_20,replace=False)
    for i in range(per_20):
        testset[z,rand[i]] =rating[z,rand[i]]
        trainset[z,rand[i]] = 0
    z =z+1
train_test_end=time.time()
print("train_test split time = ",train_test_end-train_test)
 

i_m=trainset.sum(0)/(trainset !=0).sum(0)#item mean 
np.nan_to_num(i_m,copy=False)
u_m=trainset.sum(1)/(trainset !=0).sum(1)#user-mean
#user std deviation
std_dev=np.where((trainset !=0),trainset-u_m[:,None],trainset)
std_dev=np.square(std_dev)
std_dev=std_dev.sum(1)/(trainset !=0).sum(1)
std_var=np.sqrt(std_dev)
def mod_jacc_sim(trainset): 
    t=time.time()
    x=np.where((trainset !=0),1,0)
    intr=x.dot(x.T)
    print("JACC DOT")
    den=x.sum(1)
    den1=np.broadcast_to(den,(n_users,n_users))
    den=den1*den[:,None]
    sim=intr/den
    y=time.time()-t
    print("JACCARD SIM TIEME = ",y)
    return sim
sim_jacc=mod_jacc_sim(trainset)
print("---------------------------------")
#PSS similarity
t_p=time.time()
sim_pss=np.zeros((n_users,n_users))
def sigmoid(rul,rvl) :
    return 1/(1+e(-abs(rul-rvl)))  
def proximity(rul,rvl):
  pr=1-sigmoid(rul,rvl)
  return pr
def significance(rul,rvl):
    sig=1/(1+e(-(abs(rul-rmed) * abs(rvl-rmed))))  
    return sig
def singularity(rul,rvl,i_m):
    tmp=abs(((rul+rvl)/2)-i_m)
    sign=1-1/(1+(e(-tmp)))
    return sign
def PSS(rup,rvp,i_m):
    pr=proximity(rup,rvp)
    sig=significance(rup,rvp)
    sing=singularity(rup,rvp,i_m)
    return((pr*sig*sing).sum())
#calling PSS
t_i_p=time.time()
for i in range(n_users):
    print(i)
    print(time.time()-t_i_p)
    nzi=np.nonzero(trainset[i])
    for j in range(i,n_users):
        nzj=np.nonzero(trainset[j])
        intr=np.intersect1d(nzi,nzj)
        sim_pss[i,j]=PSS(trainset[i,intr],trainset[j,intr],i_m[intr])
tmp=np.triu(sim_pss)
tmp=tmp.T+np.triu(sim_pss,k=1)
sim_pss=tmp
e_p=time.time()-t_p
print("PSS SIM TIME = ",e_p)
print("-------------------------")

#URP similarity
t_u=time.time()
tmp_urp=np.broadcast_to(u_m,(n_users,n_users))
muu_muv=abs(tmp_urp-u_m[:,None])
tmp1_urp=np.broadcast_to(std_var,(n_users,n_users))
sigma_u_v=abs(tmp1_urp-std_var[:,None])
mul_urp=muu_muv*sigma_u_v
print("MUL IN URP MU*SIG")
sim_urp=1-(1/(1+e(-mul_urp)))
e_u=time.time()-t_u
print("URP SIM TIME = ",e_u)
print("------------------------")

#NHSM similarity
JPSS=sim_pss*sim_jacc
NHSM=JPSS*sim_urp


#prediction
tdT=trainset.T
mul=(tdT).dot(NHSM)
div=np.zeros((n_movie,n_users))
stt=time.time()
for i in range(n_movie) :
    div[i] = (NHSM[tdT[i] !=0]).sum(0)
    
#np.nan_to_num(div,copy=False)
pred=(mul/div).T
np.nan_to_num(pred,copy=False)
endd=time.time()-stt
print(endd)


MAE=mean_absolute_error(testset[testset!=0],pred[testset!=0])
MSE=mean_squared_error(testset[testset!=0],pred[testset!=0])
RMSE=sqrt(MSE)

print("MAE = ",MAE)
print("RMSE = ",RMSE)

#precision,Recall,F1-measure
pred_nz=pred[testset !=0]
test_nz=testset[testset !=0]
tp=0
fp=0
fn=0
th=4 #threshold value
for i in range(len(pred_nz)):
   if test_nz[i] >=th and pred_nz[i] >=th:
       tp+=1
   elif test_nz[i] < th and pred_nz[i] >= th :
       fp+=1
   elif test_nz[i] >= th and pred_nz[i] < th :
       fn+=1
       
precision=tp/(tp+fp)
recall=tp/(tp+fn)
#f1 measure= 2*(precision *recall)/(precision +recall)
f1_measure=2*(precision *recall)/(precision +recall)















