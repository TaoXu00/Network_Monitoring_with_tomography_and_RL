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


import multi_armed_bandit
import network_topology_construction as topology
import network_tomography as tomography
class main:
    def __init__(self, time):
        self.topo= topology.network_topology(time)
        self.tomography=tomography.network_tomography()
        self.MBA=multi_armed_bandit.multi_armed_bandit(self.topo) #pass the topology

    def run_tomography(self, G, monitors):
        print("Runing network tomography....")
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
        self.MBA.Initialize(G, monitors)
        # print(f"t={t}, len_total_reward: {len(total_rewards)}")
        #total_regrets, t = self.MBA.train_llc(G, source, destination, 1000, Dict_edge_theta, Dict_edge_m, optimal_delay,
        #                                     total_rewards, t, total_regrets)
        #print(f"len total_regrets: {len(total_regrets)}")
        #avg_total_regrets = [total_regrets[i] / (i + 1) for i in range(len(total_regrets))]
        # x=[i+1 for i in range(t-1)]
        # print(x)
        '''
        plt.plot(x,avg_total_regrets)
        plt.xlabel("time slot")
        plt.ylabel("avg_regrets")
        '''
        # plt.plot(x, total_regrets)
        # plt.xlabel("time slot")
        # plt.ylabel("total_regrets")
        # plt.show()
        # plt.savefig('network_tomography.png')



mynetwork=main(2000)
G =mynetwork.creat_topology(15, 0.25)
#mynetwork.tomography_verification(G)   #here the assigned delay should be 1, place modify the topo.assign_link_delay() function
monitors=mynetwork.topo.deploy_monitor(G,2,[])
trimedG=mynetwork.topo.trimNetwrok(G, monitors)
mynetwork.run_tomography(trimedG,monitors)
mynetwork.run_MAB(G,monitors)