#LM算法

def my_Func(x,y):
    return x*0.93605953573897327-y*0.35184164840470183
    
def cal_deriv(x,y,):
    data_est_output1 = x*(0.93605953573897327+0.000001)-y*0.35184164840470183
    data_est_output2 = x*(0.93605953573897327-0.000001)-y*0.35184164840470183
    return (data_est_output1 - data_est_output2) / 0.000002


import numpy as np
import matplotlib.pyplot as plt 


tao = 10**-3
threshold_stop = 10**-15
threshold_step = 10**-15
threshold_residual = 10**-15
residual_memory = []



#construct a user function
def my_Func(params,input_data):
    global Xr
    a = params[0,0]
    # b = params[1,0]
    return np.corrcoef(input_data[:,0]*np.cos(a)-input_data[:,1]*np.sin(a),Xr)[0,1]-1




#calculating the derive of pointed parameter,whose shape is (num_data,1)
def cal_deriv(params,input_data,param_index):
    params1 = params.copy()
    params2 = params.copy()
    params1[param_index,0] += 0.000001
    params2[param_index,0] -= 0.000001
    data_est_output1 = my_Func(params1,input_data)
    data_est_output2 = my_Func(params2,input_data)
    return (data_est_output1 - data_est_output2) / 0.000002

#calculating jacobian matrix,whose shape is (num_data,num_params)
def cal_Jacobian(params,input_data):
    num_params = np.shape(params)[0]
    num_data = np.shape(input_data)[0]
    J = np.zeros((num_data,num_params))
    for i in range(0,num_params):
            J[:,i] = list(cal_deriv(params,input_data,i))
    return J

#calculating residual, whose shape is (num_data,1)
def cal_residual(params,input_data,output_data):
    data_est_output = my_Func(params,input_data)
    residual = output_data - data_est_output
    return residual



#get the init u, using equation u=tao*max(Aii)
def get_init_u(A,tao):
    m = np.shape(A)[0]
    Aii = []
    for i in range(0,m):
        Aii.append(A[i,i])
    u = tao*max(Aii)
    return u
    
#LM algorithm
def LM(num_iter,params,input_data,output_data):
    num_params = np.shape(params)[0]#the number of params
    k = 0#set the init iter count is 0
    #calculating the init residual
    residual = cal_residual(params,input_data,output_data)
    #calculating the init Jocobian matrix
    Jacobian = cal_Jacobian(params,input_data)
    
    A = Jacobian.T.dot(Jacobian)#calculating the init A
    g = Jacobian.T.dot(residual)#calculating the init gradient g
    stop = (np.linalg.norm(g, ord=np.inf) <= threshold_stop)#set the init stop
    u = get_init_u(A,tao)#set the init u
    v = 2#set the init v=2
    
    while((not stop) and (k<num_iter)):
        k+=1
        while(1):
            Hessian_LM = A + u*np.eye(num_params)#calculating Hessian matrix in LM
            step = np.linalg.inv(Hessian_LM).dot(g)#calculating the update step
            if(np.linalg.norm(step) <= threshold_step):
                stop = True
            else:
                new_params = params + step#update params using step
                new_residual = cal_residual(new_params,input_data,output_data)#get new residual using new params
                rou = (np.linalg.norm(residual)**2 - np.linalg.norm(new_residual)**2) / (step.T.dot(u*step+g))
                if rou > 0:
                    params = new_params
                    residual = new_residual
                    residual_memory.append(np.linalg.norm(residual)**2)
                    #print (np.linalg.norm(new_residual)**2)
                    Jacobian = cal_Jacobian(params,input_data)#recalculating Jacobian matrix with new params
                    A = Jacobian.T.dot(Jacobian)#recalculating A
                    g = Jacobian.T.dot(residual)#recalculating gradient g
                    stop = (np.linalg.norm(g, ord=np.inf) <= threshold_stop) or (np.linalg.norm(residual)**2 <= threshold_residual)
                    u = u*max(1/3,1-(2*rou-1)**3)
                    v = 2
                else:
                    u = u*v
                    v = 2*v
            if(rou > 0 or stop):
                break;
        
    return params
  


        
        
def main():
    params = np.zeros((2,1))
    params[0,0]=0.9360595
    params[1,0]=0.3518416
    num_data = len(Xt)
    # data_input = np.array((Xt,Yt)).T
    data_input = np.array(np.linspace(0,num_data,num_data)).reshape(num_data,1)
    data_output = np.array(Xr).T
    params[0,0]=0
    params[1,0]=0
    num_iter=100
    est_params = LM(num_iter,params,data_input,data_output)
    print(est_params)
    a_est=est_params[0,0]
    b_est=est_params[1,0]
    plt.plot(residual_memory)
    plt.show()

if __name__ == '__main__':
    main()

    
    
#LM2
# -*- coding:utf-8 -*-

import numpy as np
from numpy import matrix as mat
from matplotlib import pyplot as plt
import random
import time
 
Xr = BLE.data[:15000]
Xt = BHE.data[:15000]
Yt = BHN.data[:15000]
Xr = Xr-np.mean(Xr)
Xt = Xr-np.mean(Xt)
Yt = Xr-np.mean(Yt)

n = len(Xr)
# a1,b1,c1 = 1,3,2      # 这个是需要拟合的函数y(x) 的真实参数
a1,b1,c1 = 0.35953,0,0
# h = np.linspace(0,1,n)       # 产生包含噪声的数据
h = np.vstack((np.array(Xt),np.array(Yt))).T
# y = [np.exp(a1*i**2+b1*i+c1)+random.gauss(0,4) for i in h]
y = Xr
y = mat(y)   # 转变为矩阵形式
 
def Func(abc,iput):   # 需要拟合的函数，abc是包含三个参数的一个矩阵[[a],[b],[c]]
    a = abc[0,0]
    b = abc[1,0]
    c = abc[2,0]
    # x,y = iput[0,0],iput[0,1]
    return np.cos(a)*iput[0]-np.sin(a)*iput[1]
    # return np.exp(a*iput**2+b*iput+c)
 
def Deriv(abc,iput,n):  # 对函数求偏导
    x1 = abc.copy()
    x2 = abc.copy()
    x1[n,0] -= 0.000001
    x2[n,0] += 0.000001
    p1 = Func(x1,iput)
    p2 = Func(x2,iput)
    d = (p2-p1)*1.0/(0.000002)
    return d

    
J = mat(np.zeros((n,3)))      #雅克比矩阵
fx = mat(np.zeros((n,1)))     # f(x)  100*1  误差
fx_tmp = mat(np.zeros((n,1)))
xk = mat([[0.0],[0.0],[0.0]]) # 参数初始化
lase_mse = 0
step = 0
u,v= 1,2
conve = 100

s_time = time.clock()
while (conve):
    mse,mse_tmp = 0,0
    step += 1  
    for i in range(n):
        fx[i] =  Func(xk,h[i]) - y[0,i]    # 注意不能写成  y - Func  ,否则发散
        mse += fx[i,0]**2
        for j in range(3): 
            J[i,j] = Deriv(xk,h[i],j) # 数值求导
            
    mse /= n  # 范围约束
    H = J.T*J + u*np.eye(3)   # 3*3
    dx = -H.I * J.T*fx        # 注意这里有一个负号，和fx = Func - y的符号要对应
    xk_tmp = xk.copy()
    xk_tmp += dx
    
    for j in range(n):
        fx_tmp[i] =  Func(xk_tmp,h[i]) - y[0,i]  
        mse_tmp += fx_tmp[i,0]**2
        
    mse_tmp /= n
    q = (mse - mse_tmp)/((0.5*dx.T*(u*dx - J.T*fx))[0,0])
    if q > 0:
        s = 1.0/3.0
        v = 2
        mse = mse_tmp
        xk = xk_tmp
        temp = 1 - pow(2*q-1,3)
        if s > temp:
            u = u*s
        else:
            u = u*temp
    else:
        u = u*v
        v = 2*v
        xk = xk_tmp
    print("step = %d,abs(mse-lase_mse) = %.8f" %(step,np.abs(mse-lase_mse)))  
    if abs(mse-lase_mse)<0.001:
        e_time = time.clock()
        print(e_time-s_time)
        break
       
    lase_mse = mse  # 记录上一个 mse 的位置
    conve -= 1




print(xk)
 
z = [Func(xk,i) for i in h] #用拟合好的参数画图
plt.plot(z)
plt.plot(Xr)
plt.show()
# plt.figure(0)
# plt.scatter(h,y,s = 4)
# plt.plot(h,z,'r')
# plt.show()


####三种方法对比测试

ST = readfile()
ST[0].data = ST[0].data*(-1)
ST.filter("highpass",freq=0.1)
ST.filter("lowpass",freq=1)
r_cos,r_sin = process(ST)

##my method
r_cos,r_sin = process(ST)
def MY():
    global r_cos,r_sin
    s_time=time.clock()
    rec_rc,rec_rs = r_rebuild(r_cos,r_sin)
    angle = getangle(rec_rc,rec_rs)
    e_time = time.clock()
    print("time cost:%0.3f"%(e_time-s_time),"angle:%f"%(angle/np.pi*180))
    return angle

##coherence method
def COH():
    global BHE,BHN,BLE,BLN
    dBHE = BHE.copy().resample(1) 
    dBHN = BHN.copy().resample(1) 
    dBLE = BLE.copy().resample(1) 
    dBLN = BLN.copy().resample(1) 
    
    s_time = time.clock()
    coh = []
    coh_s = []
    for r in np.linspace(0,2*np.pi,360):
        for n in range(11):
            sum=[]
            f,Cxy = sp.signal.coherence(correction(dBHE, dBHN ,r)[0].data[n*256:(n+1)*256],dBLE.data[n*256:(n+1)*256])
            for n in range(len(f)):
                if f[n]>(1/7) and f[n]<(1/6):
                    sum.append(Cxy[n])
        coh.append(np.mean(sum))
    # for r in np.linspace((np.argmax(coh)*5-5)/180*np.pi,(np.argmax(coh)*5+5)/180*np.pi,100):
        # for n in range(11):
            # sum=[]
            # f,Cxy = sp.signal.coherence(correction(BHE, BHN ,r)[0].data[n*256:(n+1)*256],BLE.data[n*256:(n+1)*256])
            # for n in range(len(f)):
                # if f[n]>(1/7) and f[n]<(1/6):
                    # sum.append(Cxy[n])
        # coh_s.append(np.mean(sum))
    # angle_1 = np.argmax(coh)*5/180*np.pi -5/180*np.pi + np.argmax(coh_s)/10/180*np.pi
    angle_1 = np.argmax(coh)*1/180*np.pi
    e_time = time.clock()
    print("time cost:%0.3f"%(e_time-s_time),"angle:%f"%(angle_1/np.pi*180))



##LM method
def Func(abc,iput):
    a = abc[0,0]
    b = abc[1,0]
    c = abc[2,0]
    return np.corrcoef((correction(iput[:,0],iput[:,1],a)[0].data[2000:]),iput[2000:,2])[0,1]

def Func_s(abc,iput,corrm,a1):
    a = abc[0,0]
    b = abc[1,0]
    c = abc[2,0]
    return np.corrcoef((correction(iput[:,0],iput[:,1],a)[0].data[2000:]),iput[2000:,2])[0,1]+((corrm-1)*(a-a1))**2

    
def Deriv(abc,iput,n):
    x1 = abc.copy()
    x2 = abc.copy()
    x1[n,0] -= 0.000001
    x2[n,0] += 0.000001
    p1 = Func(x1,iput)
    p2 = Func(x2,iput)
    d = (p2-p1)*1.0/(0.000002)
    return d




def LM():
    global BHE,BHN,BLE
    Xr = BLE.data[:2000*50]*4
    pass
    Xt = BHE.data[:2000*50]
    pass
    Yt = BHN.data[:2000*50]
    pass
    # a1,b1,c1 = 0.35953,0,0 
    hh = np.vstack((np.array(Xt),np.array(Yt),np.array(Xr))).T
    h = hh
    # h = [0]*(int(len(hh)/500)-1)
    # h = h.reshape((4,25000,3))
    # for n in range(len(h)):
        # h[n] = hh[n*500:n*500+2000]
    # n = len(h)
    n = 1
    y = np.ones(n)
    y = mat(1)
    J = mat(np.zeros((n,3)))      #雅克比矩阵
    fx = mat(np.zeros((n,1)))     # f(x)  100*1  误差
    fx_tmp = mat(np.zeros((n,1)))
    xk = mat([[0.0],[0.0],[0.0]]) # 参数初始化
    lase_mse = 0
    step = 0
    u,v= 1,2
    conve = 100
    s_time = time.clock()
    while (conve):
        mse,mse_tmp = 0,0
        step += 1  
        for i in range(n):
            fx[i] =  Func(xk,h) - y[i]    # 注意不能写成  y - Func  ,否则发散
            mse += fx[i]**2
            for j in range(3): 
                J[i,j] = Deriv(xk,h,j)
                
        mse /= n
        H = J.T*J + np.eye(3)*float(u)   # 3*3
        dx = -H.I * J.T*fx        # 注意这里有一个负号，和fx = Func - y的符号要对应
        xk_tmp = xk.copy()
        xk_tmp += dx
        for j in range(n):
            fx_tmp[i] =  Func(xk_tmp,h) - y[i]  
            mse_tmp += fx_tmp[i]**2
            
        mse_tmp /= n
        q = (mse - mse_tmp)/((0.5*dx.T*(float(u)*dx - J.T*fx))[0,0])
        if q > 0:
            s = 1.0/3.0
            v = 2
            mse = mse_tmp
            xk = xk_tmp
            temp = 1 - pow(2*q-1,3)
            if s > temp:
                u = u*s
            else:
                u = u*temp
        else:
            u = u*v
            v = 2*v
            xk = xk_tmp
        # print("step = %d,abs(mse-lase_mse) = %.8f" %(step,np.abs(mse-lase_mse)))  
        if abs(mse-lase_mse)<0.000001:
            # e_time = time.clock()
            break
           
        lase_mse = mse  # 记录上一个 mse 的位置
        conve -= 1
    # print("time cost:%0.3f"%(e_time-s_time),"angle:%f"%(xk[0,0]/np.pi*180))
    conve = 100
    corrm = Func(xk,h)
    a1=xk[0,0]
    while (conve):
        mse,mse_tmp = 0,0
        step += 1  
        for i in range(n):
            fx[i] =  Func_s(xk,h,corrm,a1) - y[i]    # 注意不能写成  y - Func  ,否则发散
            mse += fx[i]**2
            for j in range(3): 
                J[i,j] = Deriv(xk,h,j)
                
        mse /= n
        H = J.T*J + np.eye(3)*float(u)   # 3*3
        dx = -H.I * J.T*fx        # 注意这里有一个负号，和fx = Func - y的符号要对应
        xk_tmp = xk.copy()
        xk_tmp += dx
        for j in range(n):
            fx_tmp[i] =  Func_s(xk_tmp,h,corrm,a1) - y[i]  
            mse_tmp += fx_tmp[i]**2
            
        mse_tmp /= n
        q = (mse - mse_tmp)/((0.5*dx.T*(float(u)*dx - J.T*fx))[0,0])
        if q > 0:
            s = 1.0/3.0
            v = 2
            mse = mse_tmp
            xk = xk_tmp
            temp = 1 - pow(2*q-1,3)
            if s > temp:
                u = u*s
            else:
                u = u*temp
        else:
            u = u*v
            v = 2*v
            xk = xk_tmp
        # print("step = %d,abs(mse-lase_mse) = %.8f" %(step,np.abs(mse-lase_mse)))  
        if abs(mse-lase_mse)<(1/10e14):
            e_time = time.clock()
            break
           
        lase_mse = mse  # 记录上一个 mse 的位置
        conve -= 1
    angle = float(xk[0,0])
    if angle<0:
        angle += 2*np.pi
    print("time cost:%0.3f"%(e_time-s_time),"angle:%f"%(angle/np.pi*180),"step = %d"%step)
    return angle


z = [Func(xk,i) for i in h] #用拟合好的参数画图
plt.plot(z,)
plt.plot(Xr)
plt.show()


##lv method
def LV():
    global BHE,BHN,BLE,BLN
    s_time = time.clock()
    coff_e,coff_n = [],[]
    scoff_e,scoff_n = [],[]
    for r in np.linspace(0,2*np.pi,72):
        new_bhe,new_bhn = correction(BHE, BHN ,r)
        coff_e.append(np.corrcoef(new_bhe.data[2000:],BLE.data[2000:])[0][1])
    for r in np.linspace((np.argmax(coff_e)*5-5)/180*np.pi,(np.argmax(coff_e)*5+5)/180*np.pi,100):
        new_bhe,new_bhn = correction(BHE, BHN ,r)
        scoff_e.append(np.corrcoef(new_bhe.data[2000:],BLE.data[2000:])[0][1])
    angle_e = np.argmax(coff_e)*5 -5 + np.argmax(scoff_e)*0.1
    for r in np.linspace(0,2*np.pi,72):
        new_bhe,new_bhn = correction(BHE, BHN ,r)
        coff_n.append(np.corrcoef(new_bhn.data[2000:],BLN.data[2000:])[0][1])
    for r in np.linspace((np.argmax(coff_e)*5-5)/180*np.pi,(np.argmax(coff_n)*5+5)/180*np.pi,100):
        new_bhe,new_bhn = correction(BHE, BHN ,r)
        scoff_n.append(np.corrcoef(new_bhn.data[2000:],BLN.data[2000:])[0][1])
    angle_n = np.argmax(coff_n)*5 -5 + np.argmax(scoff_n)*0.1
    angle = (angle_e+angle_n)/2
    e_time = time.clock()
    print("time cost:%0.3f"%(e_time-s_time),"angle:%f"%(angle))
    return angle/180*np.pi

ST = readfile()
ST[0].data = ST[0].data*(-1)
ST.filter("highpass",freq=0.1)
ST.trim(ST[0].stats.starttime+180,ST[0].stats.endtime-180)
r_cos,r_sin = process(ST)
LV()
LM()
MY()



ST = readfile()
ST.filter("bandpass",freqmin=0.1,freqmax=0.3)
ST.trim(ST[0].stats.starttime+180,ST[0].stats.endtime-180)
r_cos,r_sin = process(ST)
LV()
LM()
MY()


