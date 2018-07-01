#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 18:46:37 2018

@author: sanjeet
"""

import pandas as pd
import numpy as np
import time
from math import sqrt
#from numpy import random
from sklearn.metrics import mean_absolute_error,mean_squared_error
e=np.exp
init=time.time()
cols=['userId','itemId','rating','timestamp']
#reading data
read_start_time=time.time()
df=pd.read_table('ml-100k/u.data',sep='\t',header=None,names=cols,engine='python')
df.drop(columns=['timestamp'],inplace=True)
read_end_time = time.time()
print("Read time = ",read_end_time-read_start_time)


# total user and movie
n_users=df.userId.max()
n_movie=df.itemId.max()



#converting data to matrix form
mat_start=time.time()
rating=np.zeros((n_users,n_movie))
for  row in df.itertuples():
    rating[row[1]-1,row[2]-1]=row[3]
mat_end=time.time()
print("Matrix conversion time = ",mat_end-mat_start)



st=time.time()
k=5
test_f=np.zeros((5,n_users,n_movie))
train_f=np.zeros((5,n_users,n_movie))
for i in range(rating.shape[0]):
    nzi=np.nonzero(rating[i])
    np.random.shuffle(nzi[0])
   # per_20=int(len(nzi[0])/k)
    l=len(nzi[0])
    for j in range(k):
        test_f[j,i,nzi[0][int(l*j/k):int((l*(j+1))/k)]] = rating[i,nzi[0][int(l*j/k):int((l*(j+1))/k)]]

for x in range(k):
    train_f[x]=np.copy(rating)
    train_f[x] = np.where((rating !=0),rating-test_f[x],rating)

end=time.time()-st
print("5 fold split time = ",end)       
rmed = 3

maei=[]
maeu=[]
rmsei=[]
rmseu=[]
precsni=[]
precsnu=[]
reclli=[]
recllu=[]
fmi=[]
fmu=[]
gimi=[]
gpimi=[]
gimu=[]
gpimu=[]

"""
n_users=5
n_movie=4
trainset=np.array(([4,3,5,4],[5,3,0,0],[4,3,3,4],[2,1,0,0],[4,2,0,0]))
"""
for t in range(0,5):
    start_time=time.time()
    u_mean = train_f[t].sum(axis=1)/(train_f[t]!=0).sum(axis=1)
    rating_m_sub = np.where((train_f[t] !=0),train_f[t]-u_mean[:,None],train_f[t])
    sim=np.zeros((n_movie,n_movie))
    user_mean = train_f[t].sum(axis = 1)/(train_f[t]!=0).sum(axis = 1)     # mean of each user( r(u) ) where r(u,i)!=0
    rating_mean_sub = np.where((train_f[t]!=0), train_f[t] - user_mean[:,None], train_f[t]) #r(u,i) - r(u)
    sim_ac = np.zeros((n_movie,n_movie))
    print("ENTRANCE TO ITEM ")
    for i in range(0,n_movie):
        for j in range(i,n_movie):
            common_users = np.where( (train_f[t][:,i]!= 0) * (train_f[t][:,j]!=0) )[0]
            nums = (rating_mean_sub[common_users,i] * rating_mean_sub[common_users,j]).sum()
            dem1s = (rating_mean_sub[common_users,i] **2 ).sum()
            dem2s = (rating_mean_sub[common_users,j] **2 ).sum()
            sim_ac[i][j] = nums / sqrt(dem1s*dem2s + 10**-12)
    upp_tr=np.triu(sim_ac,k=1)
    upp_tr=upp_tr.T
    sim_ac=sim_ac+upp_tr
    sim_ac=np.where((sim_ac <0),0,sim_ac)
    print("----------Similarity calculated item for f =",t,"------------")
    print("Runtime: {} sec   ".format(time.time()-start_time))
    print("--------Calculating prediction------------")
    
    s_t_p=time.time()
    mulp=train_f[t].dot(sim_ac)
    divp=np.zeros((n_users,n_movie))
    for l in range(n_users) :
        #print("in prediction ",i)
        nzi=np.nonzero(train_f[t][l])
        for m in range(n_movie):
            sm=(sim_ac[m,nzi]).sum()
            divp[l,m] = sm
    pred_i=mulp/divp
    np.nan_to_num(pred_i,copy=False)

    print("----------prediction calculated item for f =",t,"------------")
    print("Runtime: {} sec   ".format(time.time()-s_t_p))
    print("--------Calculating error------------")
    MAEi=mean_absolute_error(test_f[t][test_f[t]!=0],pred_i[test_f[t]!=0])
    MSEi=mean_squared_error(test_f[t][test_f[t]!=0],pred_i[test_f[t]!=0])
    RMSEi=sqrt(MSEi)
    #precision,Recall,F1-measure
    pred_nz=pred_i[test_f[t] !=0]
    test_nz=test_f[t][test_f[t] !=0]
    tpi=0
    fpi=0
    fni=0
    th=4 #threshold value
    for x in range(0,len(pred_nz)):
       if test_nz[x] >=th and pred_nz[x] >=th:
           tpi +=1
       elif test_nz[x] < th and pred_nz[x] >= th :
           fpi +=1
       elif test_nz[x] >= th and pred_nz[x] < th :
           fni +=1
           
    precisioni=tpi/(tpi+fpi)
    recalli=tpi/(tpi+fni)
    #f1 measure= 2*(precision *recall)/(precision +recall)
    f1_measurei=2*(precisioni *recalli)/(precisioni +recalli)
    maei.append(MAEi)
    rmsei.append(RMSEi)
    precsni.append(precisioni)
    reclli.append(recalli)
    fmi.append(f1_measurei)
    
    g_i=test_f[t][test_f[t]>=th]
    g_i_p=pred_i[test_f[t]>=th]
    gim_maei=mean_absolute_error(g_i_p,g_i)
    
    gp_i=test_f[t][test_f[t] !=0]
    gp_p=pred_i[test_f[t] !=0]
    gp_i_th = gp_i[gp_p>=th]
    gp_p_th = gp_p[gp_p>=th]
    gpim_maei=mean_absolute_error(gp_p_th,gp_i_th)
    print("GIM FOR ACOS = ",gim_maei)
    print("GPIM FOR ACOS = ",gpim_maei)
    gimi.append(gim_maei)
    gpimi.append(gpim_maei)
    
    print("----USER BASED NHSM   FOR FOLD = ",t,"------------")

    st_u=time.time()
    i_m=train_f[t].sum(0) / (train_f[t]!=0).sum(0)#item mean 
    np.nan_to_num(i_m,copy=False)
    u_m=(train_f[t]).sum(1)/(train_f[t] !=0).sum(1)#user-mean
    #user std deviation
    std_dev=np.where((train_f[t] !=0),train_f[t]-u_m[:,None],train_f[t])
    std_dev=np.square(std_dev)
    std_dev=std_dev.sum(1)/(train_f[t] !=0).sum(1)
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
    sim_jacc=mod_jacc_sim(train_f[t])
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
    for i in range(0,n_users):
        nzi=np.nonzero(train_f[t][i])
        for j in range(i,n_users):
            nzj=np.nonzero(train_f[t][j])
            intr=np.intersect1d(nzi,nzj)
            sim_pss[i,j]=PSS(train_f[t][i,intr],train_f[t][j,intr],i_m[intr])
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
    
    tdT=train_f[t].T
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
    
    
    MAE=mean_absolute_error(test_f[t][test_f[t]!=0],pred[test_f[t]!=0])
    MSE=mean_squared_error(test_f[t][test_f[t]!=0],pred[test_f[t]!=0])
    RMSE=sqrt(MSE)
    
    print("MAE = ",MAE)
    print("RMSE = ",RMSE)
    
    #precision,Recall,F1-measure
    pred_nz=pred[test_f[t] !=0]
    test_nz=test_f[t][test_f[t] !=0]
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
    maeu.append(MAE)
    rmseu.append(RMSE)
    precsnu.append(precision)
    recllu.append(recall)
    fmu.append(f1_measure)
    
    #GIM , GPIM
    g_i=test_f[t][test_f[t]>=th]
    g_i_p=pred[test_f[t]>=th]
    gim_maeu=mean_absolute_error(g_i_p,g_i)
    gimu.append(gim_maeu)
    
    gp_i=test_f[t][test_f[t] !=0]
    gp_p=pred[test_f[t] !=0]
    gp_i_th = gp_i[gp_p>=th]
    gp_p_th = gp_p[gp_p>=th]
    gpim_maeu=mean_absolute_error(gp_p_th,gp_i_th)
    gpimu.append(gpim_maeu)

    
print("TOTAL RUNTIME FOR 5 TRAINSET = ",time.time()-init)

    
    
    
    
    
    
    
    
    


