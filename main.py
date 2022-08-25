import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from random import sample
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
        self.MAB=multi_armed_bandit.multi_armed_bandit(self.topo, logger_mab,self.directory,self.tomography) #pass the topology
        self.time=time
        self.plotter = plotter.plotter(self.directory)

    def run_tomography(self, G, monitors, weight):
        self.logger_main.info("Runing network tomography....")
        path_list = self.topo.getPath(G, monitors,weight)
        path_matrix = self.tomography.construct_matrix(G, path_list)
        b = self.tomography.end_to_end_measurement(G, path_list,'weight')
        m_rref, inds, uninds = self.tomography.find_basis(G, path_matrix, b)
        x, count = self.tomography.edge_delay_infercement(G, m_rref, inds, uninds)
        return x, count

    def creat_topology(self,type,n,p):
        #create a network topology
        G= self.topo.graph_Generator(type, n, p)
        return G

    def tomography_verification(self, G, weight):
        '''
        In the system configuration, we random created a topology with 100 nodes with the all link weight assigned to 1.
        :param G: the topology graph
        :return: a figure named "network_tomography_verification_node%(len(G.nodes))_link_weight1.png" will be saved to show the rate of the identified edges as the growth of the deployed monitor.
                it eventually will reach to 1 when the number of the deployed monitor is equal to the number of the edges.
        '''
        monitors_list = []
        solved_edges = []
        solved_edges_count = []
        monitor_candidate_list=[]
        for n in range(0, G.number_of_nodes()+1, 5):
            monitors = self.topo.deploy_monitor(G, n, monitor_candidate_list)
            x, count = self.run_tomography(G, monitors,weight)
            monitors_list.append(monitors)
            monitor_candidate_list=monitors
            # print(f"append monitors_list:{monitors_list}")
            solved_edges.append(x)
            solved_edges_count.append(count)
        self.plotter.plot_NT_verification_edge_computed_rate_with_monitors_increasing(G, monitors_list, solved_edges_count)

    def run_MAB(self, G, monitors):
        self.MAB.Initialize(G, monitors)
        monitor_pair_list = list(combinations(monitors, 2))
        optimal_path_dict={}
        optimal_delay_dict={}
        for monitor_pair in monitor_pair_list:
            optimal_path = nx.shortest_path(G, monitor_pair[0], monitor_pair[1], weight='delay_mean', method='dijkstra')
            optimal_delay= nx.path_weight(G, optimal_path, 'delay_mean')
            optimal_path_dict[monitor_pair]=optimal_path
            optimal_delay_dict[monitor_pair]=optimal_delay
        total_rewards_dict, selected_shortest_path, expo_count,total_mse_array,edge_exploration_during_training, average_computed_edge_num = self.MAB.train_llc(G, self.time,monitor_pair_list)

        path_dict = {}
        for path in selected_shortest_path:
            p = '-'.join(path)
            if p in path_dict:
                path_dict[p] += 1
            else:
                path_dict[p] = 1
        self.logger_main.info(f"paths are explored during the training:{path_dict}")
        return expo_count, total_mse_array, total_rewards_dict, optimal_delay, edge_exploration_during_training, average_computed_edge_num

    def MAB_with_increasing_monitors(self, G, type, node_num, p):
        '''
        In the system configuration, we random created a topology with 100 nodes.
        :param G: the topology graph
        :return: a figure named "network tomography.png" will be saved to show the rate of the identified edges will be
                 increased as the growth of the deployed monitor. it eventually will reach to 1 when the number of the deployed
                 monitor is equal to the number of the edges.
        '''

        monitors_list = []
        explored_edges_num = []
        # solved_edges_count = []
        monitor_candidate_list = []
        end_nodes=[]
        total_edge_mse_list_with_increasing_monitors = []
        total_mse_reward_list = []
        total_edge_exploration_during_training_list = []
        average_computed_edge_during_training = []
        degree_list = list(G.degree(list(G.nodes)))
        #it does not make sense to differenciate the end nodes from the internal nodes.
        for edge_degree in degree_list:
            if edge_degree[1] == 2:
                end_nodes.append(edge_degree[0])
        self.logger_main.debug(f"degree_list: {degree_list}")
        self.logger_main.debug(f"end nodes list:{end_nodes}")
        #for n in range(2, len(monitor_candidate_list) + 1, 1):
        #for n in range(2, 3, 1):
        monitors=[]
        for m_p in range(10,110, 10):
            n=int((m_p/100)*len(G.nodes))
            #self.logger_main.debug(f"m_p {m_p}")
            self.logger_main.debug(f"{n} monitors will be deployed")
            if n <= len(end_nodes):
                rest_end_nodes = [elem for elem in end_nodes if elem not in monitors]
                #self.logger_main.debug(f"rest node {rest_end_nodes}")
                select = sample(rest_end_nodes, k=n - len(monitors))
                #self.logger_main.debug(f"select {select}")
                monitors = monitors + select
                self.logger_main.info(f"Monitors are deployed in nodes: {monitors}")
            else:
                monitors = self.topo.deploy_monitor(G, n, end_nodes)
            monitors = self.topo.deploy_monitor(G, n, monitors)
            self.logger_main.info(f"deloy {n} monitors: {monitors}")
            expo_count, total_mse, total_rewards_dict, optimal_delay, edge_exploration_during_training,average_computed_edge_num = self.run_MAB(
                G, monitors)
            monitors_list.append(monitors)
            explored_edges_num.append(expo_count)
            total_edge_mse_list_with_increasing_monitors.append(total_mse)
            #total_rewards_list.append(total_rewards_dict)
            total_edge_exploration_during_training_list.append(edge_exploration_during_training)
            np_array_total_mse = np.array(total_mse)
            average_computed_edge_during_training.append(average_computed_edge_num)
            #np.savetxt("mse_with_NT_in_training_node%s.txt" %(len(G.nodes)), np_array_total_mse, delimiter=",")
            self.logger_main.info(f"{expo_count} edges has been explored")
            self.topo.draw_edge_delay_sample(G,type,node_num,p)

        #self.plotter.plot_rewards_along_with_different_monitors(total_rewards_list,optimal_delay)
        #self.plotter.plot_bar_edge_exploration_training_with_increasing_monitor(G, monitors_list, explored_edges_num)
        #self.plotter.plot_mse_with_increasing_monitor_training(total_edge_mse_list_with_increasing_monitors)
        #self.plotter.plot_edge_exporation_times_with_differrent_monitor_size(G,total_edge_exploration_during_training_list)
        self.plotter.plot_edge_computed_during_training(G, average_computed_edge_during_training)





mynetwork=main(3000)
G =mynetwork.creat_topology("Barabasi", 50, 2)
#trimedG=mynetwork.topo.trimNetwrok(G, ['4','19'])
#mynetwork.tomography_verification(G,'weight')   #here the assigned delay should be 1, place modify the topo.assign_link_delay() function
trimedG=G
mynetwork.MAB_with_increasing_monitors(trimedG,'Barabasi',50,2)

#monitors=mynetwork.topo.deploy_monitor(G,2,['4','19'])
#trimedG=G
#trimedG=mynetwork.topo.trimNetwrok(G, monitors)
#mynetwork.run_tomography(trimedG,monitors)
#mynetwork.run_MAB(trimedG,monitors,'3','47')




