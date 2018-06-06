#!/usr/bin/env python3
#coding=utf-8
import time
import re
import  urllib.error
from urllib import request

def getHtml(url):
    headers = ('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36')
    opener = request.build_opener()
    opener.addheaders = [headers]
    page = opener.open(url)
    #page = request.urlopen(request.Request(url, headers={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:53.0) Gecko/20100101 Firefox/53.0'}))
    html = page.read()
    return html


def getImg(html):
    #reg = r'src="(.+?\.jpg)" pic_ext'
    #reg = r'img class="origin_image zh-lightbox-thumb lazy" src="https://pic1.zhimg.com/(.+?\.jpg)"'
    #reg = r'data-original="(.+?\.jpg)"' \
    reg = r'data-actualsrc="(.+?\.jpg)"'
    imgre = re.compile(reg)
    html=html.decode('utf-8')
    imglist = re.findall(imgre,html)
    x = 0
    for imgurl in imglist:
        try:
            request.urlretrieve(imgurl,'/pylt/img/%s.jpg' %x)
            x+=1
            time.sleep(1)
            print('downloading picture:',x)
        except urllib.error.HTTPError as reason:
            print(reason)
    print ('download completed')


html = getHtml("https://www.zhihu.com/question/38181067/answer/99607504")
print(html.decode('utf-8'))
getImg(html)

