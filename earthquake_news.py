#!/usr/bin/env python3
#coding=utf-8
import time
import re
from urllib import error
from urllib import request
from smtplib import SMTP_SSL
from email.header import Header
from email.mime.text import MIMEText

def getHtml(url):
    headers = ('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36')
    opener = request.build_opener()
    opener.handlers = [headers]
    html = opener.open(url).read()
    opener.close()
    #print(html.decode('utf-8'))
    return html


def getData(html):
    reg = r'title="(.*地震)"'
    datare = re.compile(reg)
    html = html.decode('utf-8')
    datalist = re.findall(datare, html)
    #print(datalist)
    return datalist


def Send_Mail(mail_text):
    mail_info = {
        "from": "337859428@qq.com",
        "to": "337859428@qq.com; 389553550@qq.com",
        "hostname": "smtp.qq.com",
        "username": "337859428@qq.com",
        "authorization": "ytrjusgvtkotcaig",                #"kuictsporczwddfb",这个是测试用的小号的授权码
        "mail_subject": "earthquake_reporting",
        "mail_encoding": "utf-8"}

    recivers = [
        "337859428@qq.com",         #测试qq—mail
        "m15936363577@163.com",     #测试163
        "PenghuiW@outlook.com",     #测试outlook
        
        #"403679269@qq.com",         #刘露
        "xiongtl320@126.com",       #龙龙哥
        #"240908721@qq.com",         #罗霜
        #"775447197@qq.com",         #刘宁
        #"1065917957@qq.com",        #梁凡
        "2655732606@qq.com",        #张鑫
        "479717148@qq.com",         #李潇
                ]

    smtp = SMTP_SSL(mail_info["hostname"])
    smtp.set_debuglevel(1)
    smtp.ehlo(mail_info["hostname"])
    smtp.login(mail_info["username"], mail_info["authorization"])
    msg = MIMEText(mail_text, "plain", mail_info["mail_encoding"])
    msg["Subject"] = Header(mail_info["mail_subject"], mail_info["mail_encoding"])
    msg["from"] = mail_info["from"]
    #msg["to"] = mail_info["to"]

    for r in recivers:
        smtp.sendmail(mail_info["from"], r, msg.as_string())

    smtp.quit()

html = getHtml("http://www.csi.ac.cn/publish/main/256/100500/index.html")
data = getData(html)
print(data)
#Send_Mail(str(data[0]))
current = data

if __name__ == "__main__":
    while True:
        time.sleep(3)
        data = getData(getHtml("http://www.csi.ac.cn/publish/main/256/100500/index.html"))
        print(data[0],'\n',time.asctime(time.localtime(time.time())))
        if current != data:
            current = data
            Send_Mail(str(data[0]))
            print(time.asctime(time.localtime(time.time())))
#         if input("q退出:") == 'q':
#             break


