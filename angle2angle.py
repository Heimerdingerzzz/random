import obspy
from obspy import core
from obspy.core import UTCDateTime
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy import signal
import re
import os
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import random
import time
from numpy import matrix as mat

def readfile():
    print("please select file(s):")
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilenames()
    root.destroy()
    datastream = obspy.core.stream.Stream()
    for dir in file_path:
        print(dir)
        datastream += obspy.read(dir)
    samplerate = float(input("sample rate："))
    for tr in datastream:
        if tr.stats.sampling_rate != samplerate:
            datastream.remove(tr)
    for tr in datastream.select(component = "Z"):
        datastream.remove(tr)
    datastream.merge()
    outfile = input("output file (support png, pdf, ps, eps and svg):\n")
    # datastream.plot(size = (1200,600), outfile=outfile)
    # mypath: "C:\Users\ironman\Desktop\idea\论文\使用小波分解重构法计算地震计安装方位角\datastream.svg"
    print(datastream,"\n")
    return datastream

def process(Stream):
    namehead = re.search(r"QT\.\d+\..",Stream[0].id)[0]
    global BHE,BHN,BLE,BLN
    BHE = Stream.select(id = namehead+"BHE")[0]
    BHN = Stream.select(id = namehead+"BHN")[0]
    BLE = Stream.select(id = namehead+"BLE")[0]
    BLN = Stream.select(id = namehead+"BLN")[0]
    
    start_time_default = UTCDateTime(np.max([BHE.stats.starttime.timestamp,\
                                            BHN.stats.starttime.timestamp,\
                                            BLE.stats.starttime.timestamp,\
                                            BLN.stats.starttime.timestamp]))
    end_time_default = UTCDateTime(np.min([BHE.stats.endtime.timestamp,\
                                            BHN.stats.endtime.timestamp,\
                                            BLE.stats.endtime.timestamp,\
                                            BLN.stats.endtime.timestamp]))
    while(True):
        key = input("use default time ?(Y/N)\n")
        if key=="Y" or key=="y":
            startt = start_time_default
            endt = end_time_default
            break
        elif key=="N" or key=="n":
            startt = core.UTCDateTime(input("input start time:"))
            if startt.timestamp<start_time_default.timestamp or startt.timestamp >end_time_default.timestamp:
                startt = start_time_default
                print("warning: start time out of range. replaced with default start time")
            endt = core.UTCDateTime(input("input end time:"))
            if endt.timestamp > end_time_default.timestamp or endt.timestamp<start_time_default.timestamp:
                endt = end_time_default
                print("warning: end time out of range. replaced with default end time")
            if startt.timestamp > endt.timestamp:
                startt = start_time_default
                endt = end_time_default
                print("warning: wrong input! replaced with default time")
            break
        else :
            print("input error,try again")
    BHE.trim(startt,endt)
    BHN.trim(startt,endt)
    BLE.trim(startt,endt)
    BLN.trim(startt,endt)
    x1 = (BHE.data).tolist()
    y1 = (BHN.data).tolist()
    x2 = (BLE.data*4).tolist()
    y2 = (BLN.data*4).tolist()
    x1 = x1 - np.mean(x1)
    y1 = y1 - np.mean(y1)
    x2 = x2 - np.mean(x2)
    y2 = y2 - np.mean(y2)
    rc = [0]*len(x1)
    rs = [0]*len(x1)
    for n in range(len(x1)):
        rc[n] = (x1[n]*x2[n]+y1[n]*y2[n])/(x2[n]**2+y2[n]**2)
        rs[n] = (x1[n]*y2[n]-x2[n]*y1[n])/(x2[n]**2+y2[n]**2)
    
    return rc,rs

def pauta(t):
    flag = 1
    while(flag==1):
        std = np.std(t, ddof = 1)
        aver = np.mean(t)
        # print(std,aver)
        for n in t:
            if (abs(n - aver)) > 3*std:
                # print(n,t.index(n))
                t.remove(n)
                flag = 1
                break
            else:
                flag = 0

    return t 

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def r_rebuild(rc, rs, level=None):
    max_level = pywt.dwt_max_level(len(rc),2)
    if level == None:
        level = max_level
    elif level>max_level :
        level = max_level
        print("Decomposition level is out of range! Setting to default.")
    coeffs = pywt.wavedec(rc, 'db1', level=level)
    usecoff = [coeffs[0]]
    for n in range(len(coeffs)-1):
        usecoff.append(np.zeros(len(coeffs[n+1])))
    rec_ca = pywt.waverec(usecoff, 'db1')
    coeffs = pywt.wavedec(rs, 'db1', )
    usecoff = [coeffs[0]]
    for n in range(len(coeffs)-1):
        usecoff.append(np.zeros(len(coeffs[n+1])))
    rec_sa = pywt.waverec(usecoff, 'db1')
    return rec_ca,rec_sa


def getangle(cos_alpha,sin_alpha):
    # alpha_c = np.mean(np.arccos(cos_alpha))
    # alpha_s = np.mean(np.arcsin(sin_alpha))
    alpha_c = np.arccos(np.mean(cos_alpha))
    alpha_s = np.arcsin(np.mean(sin_alpha))
    print(alpha_c,alpha_s)
    if 0 < alpha_c < np.pi/2 and 0 < alpha_s < np.pi/2 :
        print('转动角在第一象限')
        alpha = (alpha_c+alpha_s)/2
    elif np.pi/2 < alpha_c < np.pi and 0 < alpha_s < np.pi/2 :
        print('转动角在第二象限')
        alpha = (alpha_c+np.pi-alpha_s)/2
    elif np.pi/2 < alpha_c < np.pi and -np.pi/2 < alpha_s < 0:
        print('转动角在第三象限')
        alpha = (2*np.pi-alpha_c+np.pi-alpha_s)/2
    elif 0 < alpha_c < np.pi/2 and -np.pi/2 < alpha_s < 0:
        print('转动角在第四象限')
        alpha = (2*np.pi-alpha_c+2*np.pi+alpha_s)/2
    else :
        print("error occur")
        return 0
    print(alpha*180/np.pi)
    return alpha

def correction(h_e, h_n ,angle):
    h_e_new = h_e.copy()
    h_n_new = h_n.copy()
    h_e_new.data = h_e.data*np.cos(angle) + h_n.data*np.cos(angle + np.pi/2) 
    h_n_new.data = h_e.data*np.sin(angle) + h_n.data*np.sin(angle + np.pi/2)
    return h_e_new,h_n_new

def illustration(ca,sa, level= None):
    max_level = pywt.dwt_max_level(len(ca),2)
    
    plt.figure("illustration")
    if level==None:
        level = max_level
        plt.suptitle("Decomposition level: max",size = 16, color= "blue")
        # plt.text(ca_mu-4*ca_sigma, np.max(n_1), "wavedeposition level: max",size = 5, color= "blue",\
        # bbox=dict(boxstyle="round",ec=(1., 0.5, 0.5),fc=(1., 0.8, 0.8),))
    elif level>max_level :
        level = max_level
        print("Decomposition level is out of range! Setting to default.")
        plt.suptitle("Decomposition level: "+str(level),size = 16, color= "blue")
    else:
        plt.suptitle("Decomposition level: "+str(level),size = 16, color= "blue")
        # plt.text(ca_mu-4*ca_sigma, np.max(n_1), "wavedeposition level: "+str(level), size = 5, color= "blue",\
        # bbox=dict(boxstyle="round",ec=(1., 0.5, 0.5),fc=(1., 0.8, 0.8),))
    rec_ca,rec_sa = r_rebuild(ca, sa, level=level)
    
    ca_mu = np.mean(ca)
    ca_sigma = np.std(ca, ddof =1)
    sa_mu = np.mean(sa)
    sa_sigma = np.std(sa, ddof =1)
    plt.subplot(1,2,1)
    n_1, bins_1, patches_1 = plt.hist(ca,bins = 1000,range=(ca_mu-6*ca_sigma,ca_mu+6*ca_sigma),log= True)
    plt.ylabel("Log")
    
    rec_ca_mu = np.mean(rec_ca)
    rec_ca_sigma = np.std(rec_ca, ddof =1)
    rec_sa_mu = np.mean(rec_sa)
    rec_sa_sigma = np.std(rec_sa, ddof =1)
    n_2, bins_2, patches_2 = plt.hist(rec_ca,bins = 1000,range=(ca_mu-6*ca_sigma,ca_mu+6*ca_sigma),log= True,)
    plt.legend(("Original histogram","Processed histogram "))
    
    plt.subplot(1,2,2)
    plt.plot(ca)
    plt.plot(rec_ca)
    plt.legend(("Before processing","After processing"))
    
    plt.subplots_adjust(left=0.05, right=0.98, top=0.9, bottom=0.05,wspace=0.1)
    plt.show()

ST = readfile()
ST.filter("highpass",freq=0.1)
ca,sa = process(ST)


#  angle1 = getangle(ca,sa)

plt.subplot(211)
n, bins, patches = plt.hist(ca, 1000, range=(ca_mu-3*ca_sigma,ca_mu+3*ca_sigma),color = 'g',alpha=0.75)
plt.title('Parameter Ca')
plt.subplot(212)
n, bins, patches = plt.hist(sa, 1000, range=(sa_mu-3*sa_sigma,sa_mu+3*sa_sigma),alpha=0.75)
plt.title('Parameter Sa')
plt.subplots_adjust(hspace=0.3)
plt.savefig(r"C:\Users\ironman\Desktop\idea\论文\使用小波分解重构法计算地震计安装方位角\英文投稿\figures\Ca&Sa.svg",format="svg")
# plt.savefig(r"C:\Users\ironman\Desktop\idea\论文\使用小波分解重构法计算地震计安装方位角\英文投稿\figures\Ca&Sa.jpg",format="jpg")
plt.show()

illustration(ca,sa)

rec_ca,rec_sa = r_rebuild(ca,sa)
fig,axs = plt.subplots() 
axs.grid(True)
axs.set_title("wavedec and waverec")
axs.set_ylabel("Ca")
axs.plot(ca)
axs.plot(rec_ca)
axs.legend(("Ca","rebuild Ca"),loc="best")
fig,axs = plt.subplots() 
axs.grid(True)
axs.set_title("wavedec and waverec")
axs.set_ylabel("Sa")
axs.plot(sa)
axs.plot(rec_sa)
axs.legend(("Sa","rebuild Sa"),loc="best")

angle = getangle(rec_ca,rec_sa)
#  print("未重构angle1:",angle1*180/np.pi,"重构angle:",angle*180/np.pi)
new_bhe,new_bhn = correction(BHE, BHN ,angle)

fig,axs = plt.subplots()
axs.set_title("raw data BHE & BLE")
axs.grid(True)
axs.plot((BHE.data[-1000:]-np.mean(BHE.data[-1000:])))
axs.plot(((BLE.data[-1000:]-np.mean(BLE.data[-1000:]))*4))
axs.legend(("BHE","BLE"),loc="upper left")
fig,axs = plt.subplots()
axs.set_title("raw data BHN & BLN")
axs.grid(True)
axs.plot((BHN.data[-1000:]-np.mean(BHN.data[-1000:])))
axs.plot(((BLN.data[-1000:]-np.mean(BLN.data[-1000:]))*4))
axs.legend(("BHN","BLN"),loc="upper left")

fig,axs = plt.subplots()
axs.grid(True)
axs.set_title("yield data:NEW_BHE & raw data BLE")
axs.plot((new_bhe.data[-1000:]-np.mean(new_bhe.data[-1000:])))
axs.plot(((BLE.data[-1000:]-np.mean(BLE.data[-1000:]))*4))
axs.legend(("NEW_BHE","BLE"),loc="upper left")
fig,axs = plt.subplots()
axs.grid(True)
axs.set_title("yield data:NEW_BHN & raw data BLN")
axs.plot((new_bhn.data[-1000:]-np.mean(new_bhn.data[-1000:])))
axs.plot(((BLN.data[-1000:]-np.mean(BLN.data[-1000:]))*4))
axs.legend(("NEW_BHN","BLN"),loc="upper left")

new_bhe = new_bhe.filter("bandpass",freqmin=0.1,freqmax=1)
new_bhn = new_bhn.filter("bandpass",freqmin=0.1,freqmax=1)
BLE = BLE.filter("bandpass",freqmin=0.1,freqmax=1)
BLN = BLN.filter("bandpass",freqmin=0.1,freqmax=1)


fig,axs = plt.subplots()
axs.grid(True)
axs.set_title("filtered data :NEW_BHE_flitered & BLE_filtered ")
axs.plot((new_bhe.data[-1000:]-np.mean(new_bhe.data[-1000:])),'x')
axs.plot(((BLE.data[-1000:]-np.mean(BLE.data[-1000:]))*4))
axs.legend(("NEW_BHE_FILTERED","BLE_FILTERED"),loc="upper left")
fig,axs = plt.subplots()
axs.grid(True)
axs.set_title("filtered data :NEW_BHN_flitered & BLN_filtered ")
axs.plot((new_bhn.data[-1000:]-np.mean(new_bhn.data)),'x')
axs.plot(((BLN.data[-1000:]-np.mean(BLN.data[-1000:]))*4))
axs.legend(("NEW_BHN_FILTERED","BLN_FILTERED"),loc="upper left")

plt.show()

rmse_e = np.square(np.sum((BLE.data-np.mean(BLE.data) - (BHE.data-np.mean(BHE))/4)**2)/len(BLE.data))
rmse_n = np.square(np.sum((BLN.data-np.mean(BLN.data) - (BHN.data-np.mean(BHN))/4)**2)/len(BLN.data))

rmse_e_new = np.square(np.sum((BLE.data-np.mean(BLE.data) - (new_bhe.data-np.mean(new_bhe))/4)**2)/len(BLE.data))
rmse_n_new = np.square(np.sum((BLN.data-np.mean(BLN.data) - (new_bhn.data-np.mean(new_bhn))/4)**2)/len(BLN.data))


df = pd.DataFrame(data = np.corrcoef([BLE.data, BLN.data, BHE.data, BHN.data, new_bhe.data,new_bhn.data]), \
                  columns= ["BLE", "BLN", "BHE", "BHN", "new_BHE","new_BHN"], \
                  index = ["BLE", "BLN", "BHE", "BHN", "new_BHE","new_BHN"] )
df.to_csv(r"C:\Users\ironman\Desktop\idea\论文\使用小波分解重构法计算地震计安装方位角\corrcoef.csv")

print(np.corrcoef([BLE.data, BLN.data, BHE.data, BHN.data, new_bhe.data,new_bhn.data]))
print(rmse_e, )
print(rmse_n, )
print(rmse_e_new, )
print(rmse_n_new)

