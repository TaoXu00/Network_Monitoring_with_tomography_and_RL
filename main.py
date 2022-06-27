import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg
import sympy
from numpy.random import seed
from numpy.random import randint
from numpy import random
import matplotlib.pyplot as plt
import math
import seaborn as sns
from itertools import combinations
import datetime
import multi_armed_bandit
import network_topology_construction as topology
import network_tomography as tomography
import logging
import os
class main:
    def __init__(self, time):
        basename="log"
        suffix=datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        self.directory='temp/'+"_".join([basename,suffix])+'/'
        print(f"Results saved in {self.directory}+myapp.log")
        os.makedirs(self.directory)
        logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                            datefmt='%m-%d %H:%M',
                            filename=self.directory+'myapp.log',
                            filemode='w')
        logger_topo=logging.getLogger("Network Topology Construction")
        logger_topo.setLevel(logging.DEBUG)
        logger_nt=logging.getLogger("Network Tomography")
        logger_nt.setLevel(logging.DEBUG)
        logger_mab=logging.getLogger("Multi Armed Bandits")
        logger_mab.setLevel(logging.DEBUG)
        self.logger_main=logging.getLogger("Main")
        self.logger_main.setLevel(logging.DEBUG)
        self.topo= topology.network_topology(time, logger_topo, self.directory)
        self.tomography=tomography.network_tomography(logger_nt)
        self.MAB=multi_armed_bandit.multi_armed_bandit(self.topo, logger_mab,self.directory) #pass the topology
        self.time=time
    def run_tomography(self, G, monitors):
        self.logger_main.info("Runing network tomography....")
        path_list = self.topo.getPath(G, monitors)
        path_matrix = self.tomography.construct_matrix(G, path_list)
        b = self.tomography.end_to_end_measurement(G, path_list)
        m_rref, inds, uninds = self.tomography.find_basis(G, path_matrix, b)
        x, count = self.tomography.edge_delay_infercement(G, m_rref, inds, uninds)
        return x, count

    def creat_topology(self,n,p):

        # construct a random topology and deploy 5 monitors
        G= self.topo.graph_Generator(n, p)
        return G

    def tomography_verification(self, G):
        '''
        In the system configuration, we random created a topology with 100 nodes.
        :param G: the topology graph
        :return: a figure named "network tomography.png" will be saved to show the rate of the identified edges will be
                 increased as the growth of the deployed monitor. it eventually will reach to 1 when the number of the deployed
                 monitor is equal to the number of the edges.
        '''
        monitors_list = []
        solved_edges = []
        solved_edges_count = []
        monitor_candidate_list=[]
        for n in range(0, G.number_of_nodes()+1, 5):
            monitors = self.topo.deploy_monitor(G, n, monitor_candidate_list)
            x, count = self.run_tomography(G, monitors)
            # print(f"n={n},monitors={monitors}")
            monitors_list.append(monitors)
            monitor_candidate_list=monitors
            # print(f"append monitors_list:{monitors_list}")
            solved_edges.append(x)
            solved_edges_count.append(count)
            # print(f"append solved_edges_count:{count}")
        # print(monitors_list)
        # print(solved_edges_count)
        x = [len(monitors) / len(G.nodes) for monitors in monitors_list]
        y = [edges_count / len(G.edges) for edges_count in solved_edges_count]
        print(x, y)
        plt.plot(x, y)
        plt.xlabel("% of nodes selected as monitors")
        plt.ylabel("% of solved edges")
        # plt.show()
        plt.savefig('network_tomography1.png')

    def run_MAB(self, G, monitors):

        total_rewards, total_regrets,optimal_delay= self.MAB.Initialize(G, monitors)

        monitor_pair_list = list(combinations(monitors, 2))
        for monitor_pair in monitor_pair_list:
            source=monitor_pair[0]
            destination=monitor_pair[1]
            optimal_path = nx.shortest_path(G, source, destination, weight='delay-mean', method='dijkstra')
            total_rewards, selected_shortest_path = self.MAB.train_llc(G, source, destination, self.time, optimal_delay, total_rewards, total_regrets)
            slots = int(len(selected_shortest_path)/100)
            '''
            x=np.zeros(slots,dtype=np.int64)
            y=np.zeros(slots,dtype=np.int64)
            y_p=np.zeros(slots)
            '''
            x=[]
            y=[]
            y_p=[]
            for i in range(slots):
                #print(f"i={i}")
                optimal_count=0
                sum_delay=0
                for j in range(i*100, (i+1)*100):
                    #print(f"j={j}")
                    #print(selected_shortest_path[j])
                    '''
                    if selected_shortest_path[j]==optimal_path:
                        optimal_count+=1
                        print("detected optimal path")
                    '''
                    #delay=0
                    #for edge in selected_shortest_path[j]:
                    #    delay+= G[edge[0]][edge[1]]['delay']
                    sum_delay+=total_rewards[i]
                average_delay=sum_delay/100
                x.append((i+1)*100)
                y.append(average_delay)
                #print(f"y={y}")
                #y_p.append(optimal_count/10)
            '''
            if(len(selected_shortest_path)%10!=0):
                count = 0
                print(f"slot ={slots},remainding:{(self.time -len(G.edges))%10}")
                for i in range(slots*10, len(selected_shortest_path)):
                    if selected_shortest_path[i]==optimal_path:
                        count+=1
                x.append(len(selected_shortest_path))
                y.append(count)
                y_p.append(len(selected_shortest_path))
            '''
            #print the selected shortest path during the training time
            path_dict={}
            for path in selected_shortest_path:
                p='-'.join(path)
                if p in path_dict:
                    path_dict[p]+=1
                else:
                    path_dict[p]=1
            self.logger_main.info(f"paths are explored during the training:{path_dict}")




            print(f"x={x}")
            print(f"y={y}")
            #print(f"y_p={y_p}")
            plt.figure()
            plt.plot(x, y)
            plt.xlabel("time")
            plt.ylabel("average delay every 100 seconds")
            plt.hlines(y=optimal_delay,xmin=0, xmax=len(selected_shortest_path),colors='red', linestyles='-', lw=2, label='optimal delay')
            #plt.show()
            plt.savefig(self.directory+'average_delay_every_100_seconds',format="PNG")

        #print(f"len total_regrets: {len(total_regrets)}")
        #avg_total_regrets = [total_regrets[i] / (i + 1) for i in range(len(total_regrets))]
        # x=[i+1 for i in range(t-1)]
        # print(x)
        '''
        plt.plot(x,avg_total_regrets)
        plt.xlabel("time slot")
        plt.ylabel("avg_regrets")
        '''




mynetwork=main(3000)
G =mynetwork.creat_topology(50, 0.25)
#mynetwork.tomography_verification(G)   #here the assigned delay should be 1, place modify the topo.assign_link_delay() function
monitors=mynetwork.topo.deploy_monitor(G,2,['3','47'])
trimedG=G
#trimedG=mynetwork.topo.trimNetwrok(G, monitors)
mynetwork.run_tomography(trimedG,monitors)
mynetwork.run_MAB(trimedG,monitors)
