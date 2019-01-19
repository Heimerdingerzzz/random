import obspy
import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from obspy import core
from scipy import signal



def readfile_1(dir = r"e:\地震所\computerangle\data\\",samplerate = 40.0,num = 3,):
	filename = os.listdir(dir)
	datastream = obspy.core.stream.Stream()
	for name in filename[:num]:
		path = dir+name
		print(path)
		datastream += obspy.read(path)
	for tr in datastream:
		if tr.stats.sampling_rate != samplerate:
			datastream.remove(tr)
	for tr in datastream.select(component = "Z"):
		datastream.remove(tr)
	datastream.merge()
	datastream.plot()
	print(datastream,"\n")
	return datastream

def readfile(samplerate = 40.0,):
	print("please select file(s):")
	root = tk.Tk()
	root.withdraw()
	file_path = filedialog.askopenfilenames()
	root.destroy()
	datastream = obspy.core.stream.Stream()
	for dir in file_path:
		print(dir)
		datastream += obspy.read(dir)
	for tr in datastream:
		if tr.stats.sampling_rate != samplerate:
			datastream.remove(tr)
	for tr in datastream.select(component = "Z"):
		datastream.remove(tr)
	datastream.merge()
	datastream.plot()
	print(datastream,"\n")
	return datastream



def func(datastream ,step = 100):
	st = datastream.copy()
	start_time_default = core.UTCDateTime(np.max([st[0].stats.starttime.timestamp,\
											st[1].stats.starttime.timestamp,\
											st[2].stats.starttime.timestamp,\
											st[3].stats.starttime.timestamp]))
	end_time_default = core.UTCDateTime(np.min([st[0].stats.endtime.timestamp,\
											st[1].stats.endtime.timestamp,\
											st[2].stats.endtime.timestamp,\
											st[3].stats.endtime.timestamp]))
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
	
	st.trim(starttime = startt,endtime = endt)
	# print(st)
	# st.spectrogram()
	st.filter("highpass",freq = 0.1)
	st.filter("lowpass", freq = 1)

	BHE = st.select(channel = 'BHE')[0]
	BHN = st.select(channel = 'BHN')[0]
	BLE = st.select(channel = 'BLE')[0]
	BLN = st.select(channel = 'BLN')[0]
	
	# signal.medfilt(BHE.data)
	# signal.medfilt(BHN.data)
	# signal.medfilt(BLE.data)
	# signal.medfilt(BLN.data)

	cor_h_en =[]
	cor_l_en =[]
	cor_hl_e =[]
	cor_hl_n =[]
	cor_h_eer30 = []
	cor_h_eer45 = []
	cor_h_eer60 = []
	cor_l_eer30 = []
	cor_l_eer45 = []
	cor_l_eer60 = []

	for n in range(int(len(BHE.data)/step)): 
		cor_h_en.append(np.corrcoef(BHE.data[n*step:(n+1)*step],BHN.data[n*step:(n+1)*step])[0][1])
		cor_l_en.append(np.corrcoef(BLE.data[n*step:(n+1)*step],BLN.data[n*step:(n+1)*step])[0][1])
		cor_hl_e.append(np.corrcoef(BHE.data[n*step:(n+1)*step],BLE.data[n*step:(n+1)*step]*4)[0][1])
		cor_hl_n.append(np.corrcoef(BHN.data[n*step:(n+1)*step],BLN.data[n*step:(n+1)*step]*4)[0][1])
		cor_h_eer30.append(np.corrcoef(BHE.data[n*step:(n+1)*step],\
			BHE.data[n*step:(n+1)*step]*np.cos(np.pi/6)+BHN.data[n*step:(n+1)*step]*np.sin(np.pi/6))[0][1])
		cor_h_eer45.append(np.corrcoef(BHE.data[n*step:(n+1)*step],\
			BHE.data[n*step:(n+1)*step]*np.cos(np.pi/4)+BHN.data[n*step:(n+1)*step]*np.sin(np.pi/4))[0][1])
		cor_h_eer60.append(np.corrcoef(BHE.data[n*step:(n+1)*step],\
			BHE.data[n*step:(n+1)*step]*np.cos(np.pi/3)+BHN.data[n*step:(n+1)*step]*np.sin(np.pi/3))[0][1])
		cor_l_eer30.append(np.corrcoef(BLE.data[n*step:(n+1)*step],\
			BLE.data[n*step:(n+1)*step]*np.cos(np.pi/6)+BLN.data[n*step:(n+1)*step]*np.sin(np.pi/6))[0][1])
		cor_l_eer45.append(np.corrcoef(BLE.data[n*step:(n+1)*step],\
			BLE.data[n*step:(n+1)*step]*np.cos(np.pi/4)+BLN.data[n*step:(n+1)*step]*np.sin(np.pi/4))[0][1])
		cor_l_eer60.append(np.corrcoef(BLE.data[n*step:(n+1)*step],\
			BLE.data[n*step:(n+1)*step]*np.cos(np.pi/3)+BLN.data[n*step:(n+1)*step]*np.sin(np.pi/3))[0][1])
			
			
	# print(np.mean(cor_h_en))
	# print(np.mean(cor_l_en))
	# print(np.mean(cor_hl_e))
	# print(np.mean(cor_hl_n))
	# print(np.mean(cor_h_eer30))
	# print(np.mean(cor_h_eer45))
	# print(np.mean(cor_h_eer60))
	# print(np.mean(cor_l_eer30))
	# print(np.mean(cor_l_eer45))
	# print(np.mean(cor_l_eer60))
	
	angle_h_en = np.mean(np.arccos(cor_h_en)*180/np.pi)
	angle_l_en = np.mean(np.arccos(cor_l_en)*180/np.pi)
	angle_hl_e = np.mean(np.arccos(cor_hl_e)*180/np.pi)
	angle_hl_n = np.mean(np.arccos(cor_hl_n)*180/np.pi)
	angle_h_eer30 = np.mean(np.arccos(cor_h_eer30)*180/np.pi)
	angle_h_eer45 = np.mean(np.arccos(cor_h_eer45)*180/np.pi)
	angle_h_eer60 = np.mean(np.arccos(cor_h_eer60)*180/np.pi)
	angle_l_eer30 = np.mean(np.arccos(cor_l_eer30)*180/np.pi)
	angle_l_eer45 = np.mean(np.arccos(cor_l_eer45)*180/np.pi)
	angle_l_eer60 = np.mean(np.arccos(cor_l_eer60)*180/np.pi)

	# angle_h_en1 = np.arccos(np.mean(cor_h_en))*180/np.pi
	# angle_l_en1 = np.arccos(np.mean(cor_l_en))*180/np.pi
	# angle_hl_e1 = np.arccos(np.mean(cor_hl_e))*180/np.pi
	# angle_hl_n1 = np.arccos(np.mean(cor_hl_n))*180/np.pi
	# angle_h_eer301 = np.arccos(np.mean(cor_h_eer30))*180/np.pi
	# angle_h_eer451 = np.arccos(np.mean(cor_h_eer45))*180/np.pi
	# angle_h_eer601 = np.arccos(np.mean(cor_h_eer60))*180/np.pi
	# angle_l_eer301 = np.arccos(np.mean(cor_l_eer30))*180/np.pi
	# angle_l_eer451 = np.arccos(np.mean(cor_l_eer45))*180/np.pi
	# angle_l_eer601 = np.arccos(np.mean(cor_l_eer60))*180/np.pi

	print("angle_h_en:",angle_h_en)
	print("angle_l_en:",angle_l_en)
	print("angle_hl_e:",angle_hl_e)
	print("angle_hl_n:",angle_hl_n)
	print("angle_h_eer30:",angle_h_eer30)
	print("angle_h_eer45:",angle_h_eer45)
	print("angle_h_eer60:",angle_h_eer60)
	print("angle_l_eer30:",angle_l_eer30)
	print("angle_l_eer45:",angle_l_eer45)
	print("angle_l_eer60:",angle_l_eer60)
	print(startt,endt)
	plt.subplot(411)
	plt.plot(BHE.data)
	plt.subplot(412)
	plt.plot(BHN.data)
	plt.subplot(413)
	plt.plot(BLE.data)
	plt.subplot(414)
	plt.plot(BLN.data)
	plt.figure()
	plt.subplot(211)
	plt.plot(BHE.data,label = "BHE")
	plt.plot(BLE.data*4,label = "BLE")
	plt.subplot(212)
	plt.plot(BHN.data,label = "BHN")
	plt.plot(BLN.data*4,label = "BLN")
	
	BHEc ,BHNc = correction(BHE, BHN ,angle_hl_e)
	plt.figure()
	plt.plot(BHEc)
	plt.plot(BLE.data*4)
	plt.figure()
	plt.plot(BHNc)
	plt.plot(BLN.data*4)

	plt.show()
	return 	angle_h_en,\
			angle_l_en,\
			angle_hl_e,\
			angle_hl_n,\
			angle_h_eer30,\
			angle_h_eer45,\
			angle_h_eer60,\
			angle_l_eer30,\
			angle_l_eer45,\
			angle_l_eer60,\
	

def correction(tr1, tr2 ,angle):
	tr1_new = tr1.copy()
	tr2_new = tr2.copy()
	tr1_new.data = tr1.data*np.cos(angle/180*np.pi) - tr2.data*np.sin(angle/180*np.pi) 
	tr2_new.data = tr1.data*np.sin(angle/180*np.pi) + tr2.data*np.cos(angle/180*np.pi)
	
	return tr1_new,tr2_new
	


datastream = readfile(samplerate = 50.0)
func(datastream, step = 600)
# func(datastream, core.UTCDateTime(2019,1,8,4,30),core.UTCDateTime(2019,1,8,5,00),step = 600)
# func(datastream, core.UTCDateTime(2019,1,8,5,30),core.UTCDateTime(2019,1,8,6,00),step = 600)
# func(datastream, core.UTCDateTime(2019,1,8,6,30),core.UTCDateTime(2019,1,8,7,00),step = 600)



