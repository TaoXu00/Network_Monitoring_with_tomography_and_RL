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
import plotter as plotter
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
        self.plotter = plotter.plotter(self.directory)

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

    def run_MAB(self, G, monitors, source, destination):

        self.MAB.Initialize(G, monitors)
        monitor_pair_list = list(combinations(monitors, 2))
        optimal_path = nx.shortest_path(G, source, destination, weight='delay_mean', method='dijkstra')
        optimal_delay = nx.path_weight(G, optimal_path, 'delay_mean')
        total_rewards, selected_shortest_path, expo_count,total_mse_array,edge_exploration_during_training = self.MAB.train_llc(G, self.time,monitor_pair_list,source, destination)
        slots = int(len(selected_shortest_path) / 100)
        x = []
        y = []
        y_p = []
        for i in range(slots):
            # print(f"i={i}")
            optimal_count = 0
            sum_delay = 0
            for j in range(i * 100, (i + 1) * 100):
                sum_delay += total_rewards[i]
            average_delay = sum_delay / 100
            x.append((i + 1) * 100)
            y.append(average_delay)
            # print(f"y={y}")
            # y_p.append(optimal_count/10)
        # print the selected shortest path during the training time
        path_dict = {}
        for path in selected_shortest_path:
            p = '-'.join(path)
            if p in path_dict:
                path_dict[p] += 1
            else:
                path_dict[p] = 1
        self.logger_main.info(f"paths are explored during the training:{path_dict}")

        print(f"x={x}")
        print(f"y={y}")
        # print(f"y_p={y_p}")
        plt.figure()
        plt.plot(x, y)
        plt.xlabel("time")
        plt.ylabel("average delay every 100 seconds")
        plt.hlines(y=optimal_delay, xmin=0, xmax=len(selected_shortest_path), colors='red', linestyles='-', lw=2,
                   label='optimal delay')
        # plt.show()
        plt.savefig(self.directory + 'average_delay_every_100_seconds', format="PNG")
        return expo_count, total_mse_array, total_rewards, optimal_delay, edge_exploration_during_training

    def MAB_with_increasing_monitors(self, G,stepsize):
        '''
        In the system configuration, we random created a topology with 100 nodes.
        :param G: the topology graph
        :return: a figure named "network tomography.png" will be saved to show the rate of the identified edges will be
                 increased as the growth of the deployed monitor. it eventually will reach to 1 when the number of the deployed
                 monitor is equal to the number of the edges.
        '''

        monitors_list = []
        explored_edges_num = []
        #solved_edges_count = []
        monitor_candidate_list=[]
        total_edge_mse_list_with_increasing_monitors=[]
        total_rewards_list=[]
        total_edge_exploration_during_training_list=[]
        monitors=['3','47']
        #for n in range(1,int(1/stepsize)+1):
        for n in range(2, 11, 2):
            '''
            num=int(n*stepsize*len(G.nodes))
            if num<2:
                num=2
            '''
            monitors = self.topo.deploy_monitor(G, n, monitors)
            self.logger_main.info(f"deloy {n} monitors: {monitors}")
            expo_count,total_mse, total_rewards, optimal_delay,edge_exploration_during_training=self.run_MAB(G, monitors, '3', '47')
            # print(f"n={n},monitors={monitors}")
            monitors_list.append(monitors)
            monitor_candidate_list=monitors
            # print(f"append monitors_list:{monitors_list}")
            explored_edges_num.append(expo_count)
            total_edge_mse_list_with_increasing_monitors.append(total_mse)
            total_rewards_list.append(total_rewards)
            total_edge_exploration_during_training_list.append(edge_exploration_during_training)
            #solved_edges_count.append(count)
            # print(f"append solved_edges_count:{count}")
            print(f"{expo_count} edges has been explored")
            self.logger_main.info(f"{expo_count} edges has been explored")
            self.topo.draw_edge_delay_sample(G)
        # print(monitors_list)
        # print(solved_edges_count)
        #self.plotter.plot_rewards_along_with_different_monitors(total_rewards_list,optimal_delay)
        #self.plotter.plot_bar_edge_exploration_training_with_increasing_monitor(G, monitors_list, explored_edges_num)
        #self.plotter.plot_mse_with_increasing_monitor_training(total_edge_mse_list_with_increasing_monitors)
        self.plotter.plot_edge_exporation_times_with_differrent_monitor_size(G,total_edge_exploration_during_training_list)


mynetwork=main(3000)
G =mynetwork.creat_topology(50, 0.25)
#trimedG=mynetwork.topo.trimNetwrok(G, ['4','19'])
trimedG=G
mynetwork.MAB_with_increasing_monitors(trimedG,0.1)
#mynetwork.tomography_verification(G)   #here the assigned delay should be 1, place modify the topo.assign_link_delay() function
#monitors=mynetwork.topo.deploy_monitor(G,2,['4','19'])
#trimedG=G
#trimedG=mynetwork.topo.trimNetwrok(G, monitors)
#mynetwork.run_tomography(trimedG,monitors)
#mynetwork.run_MAB(trimedG,monitors,'3','47')




