#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 22:58:15 2019

@author: abc
"""

import requests
import re
from collections import defaultdict
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import networkx as nx



class BJ:
    
    def __init__(self,url='http://www.bjsubway.com/station/zjgls/#'):
        
        self.__ulr = url
        self.__headers = {"User-Agent" : "User-Agent:Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0;"}
        self.__response=requests.get(self.__ulr,headers=self.__headers, 
                                     allow_redirects=False,
                                     verify=False).content.decode('gbk')
        
        
    def __get_all_stations_info(self):
        pattern_line_name=re.compile(r'<td colspan="\d">(.*?)</td>')
        result=pattern_line_name.findall(self.__response)
        pattern_station=re.compile(r'<tbody>[\s\S]*?</tbody>')
        result2=pattern_station.findall(self.__response)

        total_line_station={}
        station_distance={}
        distance_pattern=re.compile(r'\d+</td>')
        for line,station in zip(result,result2):
            x=station.split('——')
            distance=distance_pattern.findall(station)
            distance=[x.split('</td>')[0] for x in distance]
            if len(line)<=13:
                line=line[:3]
            else:
                line=line[:4]
            total_line_station[line]=[i.split('<th>')[-1] for i in  x]
            total_line_station[line][-1]=total_line_station[line][-1].split('</th')[0]
            for i in range(0,len(total_line_station[line])-1):
                station_distance[total_line_station[line][i]]=\
                    [total_line_station[line][i+1]+' '+distance[i]]
                    
        return total_line_station,station_distance
    
    def __get_location(self,location):
        pattern=re.compile(r'(\w+),(\d+.\d+),(\d+.\d+)')
        result=pattern.findall(location)
        graph={}
        for value in result:
            graph[value[0]]=(float(value[1]),float(value[-1]))
        return graph
    
    
    def geo_distance(self,origin, destination):

        lat1, lon1 = origin
        lat2, lon2 = destination
        radius = 6371  # km
        # 转换为弧度返回
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dlon / 2) * math.sin(dlon / 2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        d = radius * c

        return d
    
    def search(self,A,B):
        pass
    
    def main(self):
        total_line_station,station_distance=self.__get_all_stations_info()
        graph=self.__get_location(location)
        return total_line_station,station_distance,graph
        
if __name__=='__main__':
    bj=BJ()
    total_line_station,station_distance,graph=bj.main()
    bj_station_graph = nx.Graph()
    bj_station_graph.add_nodes_from(list(graph.keys()))
    nx.draw(bj_station_graph, graph, with_labels=True, node_size=30)