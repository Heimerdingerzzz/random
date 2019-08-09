ST = readfile()
ST[0].data = ST[0].data*(-1)
r_cos,r_sin = process(ST)
ST.filter("highpass",freq = 0.1)

s = time.clock()
rec_rc,rec_rs = r_rebuild(r_cos,r_sin)
angle = getangle(rec_rc,rec_rs)
e = time.clock()
print(e-s)
new_bhe,new_bhn = correction(BHE, BHN ,angle)

s = time.clock()
coff = [] 
coff_s = []
for r in np.linspace(0,2*np.pi,360):
    coff.append(np.corrcoef([correction(BHE, BHN ,r)[0].data,BLE.data])[0][1])

for r in np.linspace((np.argmax(coff)*5-5)/180*np.pi,(np.argmax(coff)*5+5)/180*np.pi,100):
    coff_s.append(np.corrcoef([correction(BHE, BHN ,r)[0].data,BLE.data])[0][1])

angle_1 = np.argmax(coff)*5/180*np.pi -5/180*np.pi + np.argmax(coff_s)/10/180*np.pi
e = time.clock()
print(e-s)
print(angle_1*180/np.pi)

plt.figure()
plt.plot(coff)
plt.plot(np.argmax(coff),np.max(coff),'.')
# plt.annotate("("+str(np.argmax(coff))+", "+str(np.around(np.max(coff),decimals=3))+")", xy = (np.argmax(coff)-1, 0.85),)
plt.annotate("(20.4, 0.976)", xy = (17, 0.8),)
plt.grid(True)
plt.xlim((0,360))
plt.xlabel("Angle(°)")
plt.ylabel("Correlation coefficient")



angle_1 = np.argmax(coff)/180*np.pi -5/180*np.pi + np.argmax(coff)/180*np.pi
print(angle_1)
new_bhe_1,new_bhn_1 = correction(BHE, BHN ,angle_1)




#三种方法对比图
angle = LV()
new_bhe,new_bhn = correction(BHE, BHN ,angle)
angle_1 = LM()
new_bhe_1,new_bhn_1 = correction(BHE, BHN ,angle_1)
angle_2 = MY()
new_bhe_2,new_bhn_2 = correction(BHE, BHN ,angle_2)

fig,axs = plt.subplots(4,1)
axs[0].grid(True)
axs[1].grid(True)
axs[2].grid(True)
axs[3].grid(True)
axs[0].set_ylabel("Counts")
axs[1].set_ylabel("Counts")
axs[2].set_ylabel("Counts")
axs[3].set_ylabel("Counts")
axs[0].plot((new_bhe.data[-500:]-np.mean(new_bhe.data[-500:])),color = 'B')
axs[1].plot((new_bhe_1.data[-500:]-np.mean(new_bhe_1.data[-500:])),color = 'B')
axs[2].plot((new_bhe_2.data[-500:]-np.mean(new_bhe_2.data[-500:])),color = 'B')
axs[3].plot(((BLE.data[-500:]-np.mean(BLE.data[-500:]))*4),color = 'G')
axs[0].legend(["NEW_BHE result of Lv's method"],loc="best")
axs[1].legend(["NEW_BHE result of Ringler's method"],loc="best")
axs[2].legend(["NEW_BHE result of Our method"],loc="best")
axs[3].legend(["BLE in four times magnification "],loc="best")

fig,axs = plt.subplots(4,1)
axs[0].grid(True)
axs[1].grid(True)
axs[2].grid(True)
axs[3].grid(True)
axs[0].set_ylabel("Counts")
axs[1].set_ylabel("Counts")
axs[2].set_ylabel("Counts")
axs[3].set_ylabel("Counts")
axs[0].plot((new_bhn.data[-500:]-np.mean(new_bhn.data[-500:])),color = 'B')
axs[1].plot((new_bhn_1.data[-500:]-np.mean(new_bhn_1.data[-500:])),color = 'B')
axs[2].plot((new_bhn_2.data[-500:]-np.mean(new_bhn_2.data[-500:])),color = 'B')
axs[3].plot(((BLN.data[-500:]-np.mean(BLN.data[-500:]))*4),color = 'G')
axs[0].legend([r"NEW_BHN result of Lv's method"],loc="lower left")
axs[1].legend([r"NEW_BHN result of Ringler's method"],loc="lower left")
axs[2].legend([r"NEW_BHN result of Our method"],loc="lower left")
axs[3].legend(["BLN in four times magnification"],loc="lower left")
plt.show()

