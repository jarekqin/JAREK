#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 20:11:25 2019

@author: abc
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import random
import re

url=' https://baike.baidu.com/item/%E5%8C%97%E4%BA%AC%E5%9C%B0%E9%93%81/408485'
headers = {"User-Agent" : "User-Agent:Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0;"}
response=requests.get(url,headers=headers, allow_redirects=False).content.decode('utf8')
pattern=re.compile(r'<a target=_blank href="(.*?)">北京地铁(.*?)</a></td><td')
result=pattern.findall(response)
header='http://baike.baidu.com'
total_url=dict()
for add in result:
    if len(add[-1])>8:
        continue
    total_url[add[-1]]=header+add[0]

url=list(total_url.values())[0]
headers = {"User-Agent" : "User-Agent:Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0;"}
response_7=requests.get(url,headers=headers).content.decode('utf8')
pattern_7_1=re.compile(r'<a target=_blank href=\"(.*?)\">(.*?)</a>')
result_7_1=pattern_7_1.findall(response_7)

test=[]
for station in result_7_1:
    if '站' in station[-1].split('>')[-1]:
        test.append(station[-1].split('>')[-1])
test=list(set(test))
test[-1]='施园站'