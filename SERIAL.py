import serial
import numpy as np
import re
    
def use_ser():
    com = 'COM%s'%input('Select COM Port ：')
    baud = int(input('Seting Baudrate:'))
    print(com, baud)
    ser = serial.Serial(com, baud)      #initiation，COM口与波特率
    while True:
        data = ser.read(16).decode()
        #f = re.search(r'u(.*)', data).span()[0]
        #data = data[:f]
        print(data)
        '''
        if data[-3] == "V" :
            data = data[:-4]
            if "\n" in data:
                print (float(data[3:]))
            else:
                print(float(data))
        '''
    ser.close()                         #串口关
    


def data_t():
    pass

    
use_ser()
