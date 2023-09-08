import pickle
import sys
import numpy as np
import os
import scipy.signal as sg
import wfdb
from tqdm import tqdm
import numpy as np
from keras import backend as K
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
from scipy.signal import resample
import glob
import pywt
import pandas as pd
import neurokit2 as nk
import csv
from collections import Counter
import warnings
from IPython.display import display
import shutil
import posixpath
import pyhrv
import json
from vectorizedsampleentropy import vectsampen as vse
import math
import heartpy as hp
from statsmodels.tsa.ar_model import AR
import pyhrv.tools as tools
import glob
from sklearn.metrics import r2_score
import collections
from math import sqrt
from sklearn.metrics import mean_squared_error
import scipy.signal
from biosppy.signals import ecg
import gc
import bz2
gc.enable()
pd.set_option('display.max_colwidth', None)
fs = 125
base_dir = "dataset"
sampling_rate = 360
invalid_labels = ['|', '~', '!', '+', '[', ']', '"', 'x']  # non-beat labels
before = 90
after = 110
tol = 0.05
cpuCount = os.cpu_count() 
import psutil
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# setting  HRV functuon        
def estimate_shannon_entropy(dna_sequence):
            m = len(dna_sequence)
            bases = collections.Counter([tmp_base for tmp_base in dna_sequence])
 
            shannon_entropy_value = 0
            for base in bases:
        # number of residues
                n_i = bases[base]
        # n_i (# residues type i) / M (# residues in column)
                p_i = n_i / float(m)
                entropy_i = p_i * (math.log(p_i, 2))
                shannon_entropy_value += entropy_i
 
            return shannon_entropy_value * -1
			

def poincare_sd2(rr):
    rr_n = rr[:-1]
    rr_n1 = rr[1:]

    sd1 = np.sqrt(0.5) * np.std(rr_n1 - rr_n)
    sd2 = np.sqrt(0.5) * np.std(rr_n1 + rr_n)

    m = np.mean(rr)
    min_rr = np.min(rr)
    max_rr = np.max(rr)
    
    
    return sd2


def hrvtransform_filter_rr(rr_normal,fs,hrv1,settings_time,settings_welch,settings_ar,settings_lomb,settings_nonlinear):



#         working_data, measures = hp.process(hrdata1, fs)
#hp.plotter(working_data, measures)


        nni = np.array(rr_normal)
        nni_old = nni
        # nni filter in hr 50~150 
        hr = 60/(nni/1000)
        ind0=np.where((hr >= 50 ) &  (hr <= 150))
#         normal_hr_filter_in50_150=np.array(hr)[ind0]
        
        #update nni  with  50<= HR  <= 150
        nni = np.array(nni)[ind0]
#         total_time_proportion=sum(nni)/sum(rri_0)
#         number_proportion=len(nni)/len(rri_0)
        
        hr = 60/(nni/1000)
#  Mean.rate
        meanrate= np.mean(hr)

# Poincar..SD2
        sd2=poincare_sd2(nni)


   # Compute the pyHRV parameters
        results = pyhrv.hrv(nni=nni,
                       kwargs_time=settings_time,
                       kwargs_welch=settings_welch,
                       kwargs_ar=settings_ar,
                       kwargs_lomb=settings_lomb,
                       kwargs_nonlinear=settings_nonlinear)





#DFA.Alpha.1
        DFA_Alpha1 = results['dfa_alpha1']





#LF.HF.ratio.LombScargle
        ratio=results['lomb_ratio']

        nni0=np.array(nni)/1000

        nni_diff=np.diff(nni0)
        nni_rmfirst=nni0[1:] 

# aFdP
# RR allan factor  distance

        aFdP = np.var(nni_diff)/(2*np.mean(nni_rmfirst) ) -1



# fFdP
# RR fano factor distance
        fano_rr = np.var(nni0)/np.mean(nni0)
        fFdP =fano_rr-1

        ax = pd.plotting.autocorrelation_plot(nni0)
        c0=ax.lines[5].get_data()[1]
        arr = np.array(c0)

        df = pd.DataFrame(data=nni0)
        df.columns =['rr']
        df['time'] =np.cumsum(df['rr'])
        columns_titles = ["time","rr"]
        df=df.reindex(columns=columns_titles)
        df['stationary']=df['rr'].diff(arr.argmax(0))



#create datasets
        X = df['stationary'].dropna()
# the autoregression model
        model = AR(X)
        model_fitted = model.fit()
        predictions = model_fitted.predict()

        r2 = r2_score(df['stationary'].tail(len(predictions)), predictions)
        rmse = sqrt(mean_squared_error(df['stationary'].tail(len(predictions)), predictions))

        df['stationary']=df['rr'].diff(model_fitted.k_ar)
#create datasets
        X = df['stationary'].dropna()

#train the autoregression model
        model = AR(X)
        model_fitted = model.fit()
        predictions = model_fitted.predict()
        r2 = r2_score(df['stationary'].tail(len(predictions)), predictions)
        rmse = sqrt(mean_squared_error(df['stationary'].tail(len(predictions)), predictions))
# Aerr
        ARerr = rmse 


        L = nni
        r = 0.2*np.std(L)
        m = 1
        QSE=vse.qse(L, m, r) 

        shannEn=estimate_shannon_entropy(nni0)




#======================================================================================================================
    
        ratio=float(ratio)
        DFA_Alpha1=float(DFA_Alpha1)
        sd2=sd2/1000
   


        hrvvar = np.asarray([aFdP,fFdP,ARerr,DFA_Alpha1,meanrate,sd2,shannEn,ratio])


        hrv0 = pd.DataFrame(hrvvar)
#hrv0
        hrv0=hrv0.transpose()

        hrv1=hrv1.append(hrv0)
        hrv1.columns=['aFdP', 'fFdP', 'ARerr', 'DFA.Alpha.1', 'Mean.rate','Poincar..SD2', 'shannEn', 'LF.HF.ratio.LombScargle']
        
        
        plt.close('all')
        
        df1 = pd.DataFrame(hrv1)
        a0 = df1.reset_index(drop=True)
        a0 = pd.DataFrame(a0)
        finalhrv = a0.drop([0])
        for i in finalhrv.columns:
                 finalhrv[i] = pd.to_numeric(finalhrv[i])
        return finalhrv