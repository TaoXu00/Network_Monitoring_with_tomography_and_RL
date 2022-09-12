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
import network_topology_construction as topo
from itertools import combinations
import plotter as plotter
class multi_armed_bandit:
    # main(100, 0.5)
    def __init__(self, topo, logger, directory):
        self.topo=topo
        self.plotter=plotter.plotter(directory)
        self.Dict_monitor_path={}
        self.Dict_path_theta={}   #the observed actual average delay-mean for each edge
        self.Dict_path_m = {}       #the counter for how many times this edge has been observed
        self.t=1
        self.logger=logger
        self.directory=directory
        self.edge_delay_difference_list=[]
        self.edge_exploration_times=[]

    def Initialize(self, G, monitors):
        '''
        :param G: The network topology
        :param source: the source node
        :param destination:  the destination node
        :param Dict_edge_scales: the vector used to construct the delay exponential distribution
        :param optimal_delay: the delay computed with mean vector
        :return: Dict_edge_theta: updated real mean delay vector , Dict_edge_m: updated vector for tracking how many times this link has been visited so far,
                         t - the timestamp, total_rewards - accumulate rewards, total_regrets -accumulate regrets
        '''
        self.logger.info("Multi Armed Bandits with policy UBC1 Initializing..........")
        monitor_pair_list = list(combinations(monitors, 2))
        print(monitor_pair_list)
        for monitor_pair in monitor_pair_list:
            self.Dict_monitor_path[monitor_pair]=[]
        for monitor_pair in monitor_pair_list:
            self.logger.debug(f"collecting paths between m{monitor_pair[0]} and m{monitor_pair[1]}")
            cycle=0
            while(len(self.Dict_monitor_path[monitor_pair])<20):
                path=nx.shortest_path(G,monitor_pair[0], monitor_pair[1],"weight")
                path_pair = self.construct_pathPair_from_path(G, path)
                for edge in path_pair:
                    incresed = randint(0, 10)
                    G[edge[0]][edge[1]]['weight'] += incresed
                #self.logger.debug("selected shortest path: %s" % (path))
                if path not in self.Dict_monitor_path[monitor_pair]:
                    cycle=0
                    #self.logger.debug("put selected shortest path: %s" % (path))
                    self.Dict_monitor_path[monitor_pair].append(path)
                    id="-".join(path)
                    delay=nx.path_weight(G, path, "delay")
                    self.Dict_path_theta[id] = delay
                    self.Dict_path_m[id] = 1
                else:
                    cycle+=1
                    if cycle==200:
                        break
            self.logger.debug("collected %d paths for %s and %s "  %(len(self.Dict_monitor_path[monitor_pair]), monitor_pair[0], monitor_pair[1]))

            self.topo.assign_link_delay(G)
            self.t = self.t + 1
        self.logger.info(
            "===============================Initialization is finished======================================================== ")
        #self.logger.debug("Dict_monitor_path: %s" %(self.Dict_monitor_path))
        self.logger.debug("Dict_path_m: %s" %(self.Dict_path_m))
        self.logger.debug("Dict_path_theta: %s" %(self.Dict_path_theta))


    def compute_path_delay_with_path_pair(self, G, pathpair):
        delay = 0
        for edge in pathpair:
            delay += G[edge[0]][edge[1]]['delay']
        return delay



    def construct_pathPair_from_path(self,G, path):
        pathpair = []
        for i in range(len(path) - 1):
            if (path[i], path[i + 1]) in list(G.edges):
                pathpair.append((path[i], path[i + 1]))
            else:
                pathpair.append((path[i + 1], path[i]))
        return pathpair

    def update_MBA_variabels(self, G, explored_path_list):   ##different
        for path in explored_path_list:
            path_delay=nx.path_weight(G,path,"delay")
            path_id="-".join(path)
            self.Dict_path_theta[path_id] = (self.Dict_path_theta[path_id]* self.Dict_path_m[path_id]+ path_delay)/(self.Dict_path_m[path_id]+1)
        self.Dict_path_m[path_id]=self.Dict_path_m[path_id]+1

    def optimal_path(self, G, monitor_pair_list):
        optimal_delay_dict = {}
        optimal_path_dict={}
        for monitor_pair in monitor_pair_list:
            optimal_path = nx.shortest_path(G, source=monitor_pair[0], target=monitor_pair[1], weight='delay_mean', method='dijkstra')
            self.logger.info("optimal path: %s" %(optimal_path))
            optimal_delay=0
            for i in range(len(optimal_path) - 1):
                optimal_delay += G[optimal_path[i]][optimal_path[i + 1]]["delay_mean"]
            optimal_delay_dict[monitor_pair]=optimal_delay
            optimal_path_dict[monitor_pair]=optimal_path
        self.logger.info("optimal_delay: %s" %(optimal_delay_dict))
        return optimal_delay_dict,optimal_path_dict

    def edges_of_optimal_path(self, optimal_path_dict):
        optimal_path_list=[]
        for key in optimal_path_dict:
            optimal_path_list.append(optimal_path_dict[key])
        optimal_edge_set = []
        for path in optimal_path_list:
            pathpair=self.construct_pathPair_from_path(path)
            for edge in pathpair:
                if edge not in optimal_edge_set:
                    optimal_edge_set.append(edge)
        return optimal_edge_set

    def LLC_policy(self, G, monitor_pair):
        # select a path which solves the minimization problem
        path_list=self.Dict_monitor_path[monitor_pair]
        minimum_ubc_factor=math.inf
        selected_path=[]
        for path in path_list:
            path_id= "-".join(path)
            ubc1_factor=self.Dict_path_theta[path_id]-math.sqrt(2*self.t/self.Dict_path_m[path_id])
            if ubc1_factor < minimum_ubc_factor:
                minimum_ubc_factor=ubc1_factor
                selected_path=path
        return selected_path


    def end_to_end_measurement(self, G, path_list):
        #print("pathlist= %s" %(path_list))
        path_delays = []
        average_edge_delay_list=[]
        for path in path_list:
            #print("path: %s" %(path))
            path_delay = 0
            for edge in path:
                path_delay = path_delay + G[edge[0]][edge[1]]['delay']
            path_delays.append(path_delay)
            average_edge_delay=path_delay/len(path)
            average_edge_delay_list.append(average_edge_delay)
        b = np.array([path_delays])  # the delay of the selected path
        return b, average_edge_delay_list

    def train_llc(self,G, time, monitor_pair_list):
        optimal_delay_dict,optimal_path_dict= self.optimal_path(G, monitor_pair_list)
        selected_shortest_path=[]
        total_rewards_dict = {}   #in the current implementation, it is for only one pair of monitors
        total_regrets = []
        computed_edge_num=[]
        correct_shortest_path_selected_rate=[]
        rewards_mse=[]
        ## check the delay difference of  edges in the optimal path before training
        self.logger.debug("start trainning...")
        for monitor_pair in monitor_pair_list:
            total_rewards_dict[monitor_pair]=[]
        for i in range(time-self.t):
            self.logger.info("t= %s" %(self.t))
            explored_path_list = []
            num_correct_shortest_path=0
            for monitor_pair in monitor_pair_list:
                shortest_path=self.LLC_policy(G, monitor_pair)
                self.logger.debug("shortest_path: %s, optimal path: %s" % (shortest_path, optimal_path_dict[monitor_pair]))
                if shortest_path == optimal_path_dict[monitor_pair]:
                    num_correct_shortest_path+=1
                explored_path_list.append(shortest_path)
                rewards = nx.path_weight(G, shortest_path, 'delay')
                total_rewards_dict[monitor_pair].append(rewards)
            self.logger.info("selected shortest path: %s" %(explored_path_list))
            correct_shortest_path_selected_rate.append(num_correct_shortest_path/len(monitor_pair_list))
            self.update_MBA_variabels(G, explored_path_list)
            self.t = self.t + 1  # the time slot increase 1
            self.topo.assign_link_delay(G)
        rewards_mse_list=self.compute_rewards_mse(total_rewards_dict, optimal_delay_dict)
        average_regret_list=self.compute_averaged_regret(total_rewards_dict, optimal_delay_dict)
        self.plotter.plot_time_average_rewards(rewards_mse_list)
        self.plotter.plot_average_regrets(average_regret_list)
        self.plotter.plot_rate_of_correct_shortest_path(correct_shortest_path_selected_rate)  #implement this function

        self.logger.debug("training is finished")
        #self.logger.debug("Dict_edge_m values are added to the edge_exploration_times array")
        return rewards_mse_list,selected_shortest_path

    def compute_rewards_mse(self,total_rewards_dict, optimal_delay_dict):
        key_list = list(total_rewards_dict.keys())
        rewards_mse_list = []
        sum_rewards_Dict = {}
        time_average_rewards_Dict = {}
        for key in key_list:
            sum_rewards_Dict[key] = 0
            time_average_rewards_Dict[key] = 0
        for i in range(len(total_rewards_dict[key_list[0]])):
            sum_square = 0
            for key in key_list:
                sum_rewards_Dict[key] += total_rewards_dict[key][i]
                time_average_rewards_Dict[key] = sum_rewards_Dict[key] /(i + 1)
                sum_square += (time_average_rewards_Dict[key] - optimal_delay_dict[key]) ** 2
            rewards_mse_list.append(sum_square /len(key_list))
        return rewards_mse_list

    def compute_averaged_regret(self, total_rewards_dict, optimal_delay_dict):
        key_list = list(total_rewards_dict.keys())
        sum_rewards_Dict = {}
        average_regret_list = []
        for key in key_list:
            sum_rewards_Dict[key] = 0
            #time_average_rewards_Dict[key] = 0
        for i in range(len(total_rewards_dict[key_list[0]])):
            regret_list = []
            for key in key_list:
                sum_rewards_Dict[key] += total_rewards_dict[key][i]
                regret= sum_rewards_Dict[key] - (i+1)*optimal_delay_dict[key]
                #self.logger.info("regret: %f" %(regret))
                regret_list.append(regret)
            #self.logger.info("regret_list: %s: " %regret_list)
            average_regret_list.append(sum(regret_list)/math.log(i+2))
            #time_averaged_regret_list.append(sum(regret_list)/len(key_list)/(i+1))
            self.logger.info("average_regret_list: %s" %average_regret_list)
        return average_regret_list











