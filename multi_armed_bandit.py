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
        self.Dict_edge_theta = {}   #the observed actual average delay-mean for each edge
        self.Dict_edge_m = {}       #the counter for how many times this edge has been observed
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
        for edge in G.edges:
            self.Dict_edge_theta[edge] = 0
        for edge in G.edges:
            self.Dict_edge_m[edge] = 0

        #total_rewards = []   #in the current implementation, it is for only one pair of monitors
        #total_regrets = []
        sample_delays = []

        monitor_pair_list=list(combinations(monitors, 2))
        #optimal_delay, optimal_path = self.optimal_path(G, source, destination)
        self.logger.info("Multi Armed Bandits Initializing..........")
        for edge_G in G.edges:   #traversal each link to guarantee every link is covered at least once
            n1 = edge_G[0]
            n2 = edge_G[1]
            #self.logger.debug(f"t= {self.t} initializing edge {n1}, {n2}")
            for monitor_pair in monitor_pair_list:
               #self.logger.debug(f"check with the monitor_pair: {monitor_pair}")
                m1 = monitor_pair[0]
                m2 = monitor_pair[1]
                   #don't know why we need to find the optimal delay
                # find the shortest path between source to n1 and n2 to destination
                '''store the sampled delay for each link'''
                delays = []
                for edge in G.edges:
                    delays.append(G[edge[0]][edge[1]]['delay'])
                # print(delays)
                sample_delays.append(delays)
                # index of time slot
                found=self.find_path(G, edge_G, n1, n2, m1, m2)
                if found == 1:  # does not find the path between this monitor pair
                    #self.logger.debug(f"found the path with monitor1 {m1} monitor2 {m2} and left_node {n1} right_node {n2}")
                    break
                elif found ==2:
                    #self.logger.debug(f"return 2 in Check1")
                    continue
                elif found==3:   #the right node is the same as the source node, now check the edge (n2, n1)
                    #self.logger.debug("return 3 in check 1")
                    found_inverse=self.find_path(G, edge_G, n2, n1, m1, m2)
                    if found_inverse==2:
                        #self.logger.debug("return 2 in check 2")
                        continue
                    elif found_inverse==1:
                        #self.logger.debug(f"found the path with monitor {m1} monitor {m2} and left_node {n2} right_node {n1}")
                        break


            #total_rewards.append(rewards)
            #self.logger.debug(f"total_rewards: {total_rewards}")
            #regret = sum(total_rewards) - self.t * optimal_delay
            #total_regrets.append(regret)
            #self.logger.debug(f"regret: {regret}")
            #self.logger.debug(f"total_regrets: {total_regrets}")
            #self.logger.debug(f"t={self.t}, initialization")
            self.topo.assign_link_delay(G)
            self.t = self.t + 1

        self.logger.info(
            "===============================Initialization is finished======================================================== ")
        self.logger.debug(f"Dict_edge_m: {self.Dict_edge_m}")
        #self.logger.debug(f"Dict_edge_theta[edge]: {self.Dict_edge_theta}")
        self.plotter.plot_edge_delay_difference(G, self.Dict_edge_theta)
        delay_difference=[]
        for edge in G.edges:
            delay_difference.append(abs(self.Dict_edge_theta[edge]-G[edge[0]][edge[1]]['delay_mean']))
        self.edge_delay_difference_list.append(delay_difference)
        plt.savefig(self.directory + 'delay difference from mean after initialization', format="PNG", dpi=300, bbox_inches='tight')
        self.plotter.plot_edge_exploitation_times_bar('After initialization',self.Dict_edge_m)
        self.edge_exploration_times.append([v for k, v in self.Dict_edge_m.items() ])
        self.logger.debug(f"Dict_edge_m values are added to the edge_exploration_times array")
        self.logger.debug(f"{self.edge_exploration_times}")
        plt.savefig(self.directory + '# of edge exploration after initialization ', format="PNG", dpi=300,
                    bbox_inches='tight')

        #return total_rewards, total_regrets, optimal_delay

    def find_path(self, G, edge_G, left_node, right_node, source, destination):
        G_l=G.copy()
        G_l.remove_node(right_node)
        if(destination in G_l.nodes):
            G_l.remove_node(destination)
        #if(source not in G_l.nodes or left_node not in G_l.nodes):
        #    return 0
        if(source in G_l.nodes and left_node  in G_l.nodes):
            try:
                shortest_path_l = nx.shortest_path(G_l, source=source, target=left_node, weight='weight',
                                                   method='dijkstra')
                #self.logger.debug(f"shortest path from {source} to {left_node}: {shortest_path_l}")
                pathpair_list_l = self.construct_pathPair_from_path(shortest_path_l)
                G_r = G.copy()
                # print(G_r.nodes)
                for edge in pathpair_list_l:
                    if edge in G_r.edges:
                        G_r.remove_edge(edge[0], edge[1])
                    else:
                        G_r.remove_edge(edge[1], edge[0])
            except Exception as e:
                #self.logger.error(str(e)+"occurred, try the inversed direction.")
                return 3
            G_r.remove_node(left_node)
            if(source in G_r.nodes):
                G_r.remove_node(source)
            if (destination in G_r.nodes and right_node in G_r.nodes):
                try:
                    shortest_path_r = nx.shortest_path(G_r, source=right_node, target=destination, weight='weight',
                                               method='dijkstra')
                    #self.logger.debug(f"shortest path from {right_node} to {destination}: {shortest_path_r}")
                    pathpair_list_r = self.construct_pathPair_from_path(shortest_path_r)
                    pathpair_list_l.append(edge_G)
                    pathpair_list = pathpair_list_l + pathpair_list_r
                    if (len(pathpair_list) != 0):
                       # self.logger.debug(
                       #     f"The MAB variables are updated for edge {pathpair_list}!")  # it works for the first edge, check why it does not go to the for loop.
                        self.update_MBA_variabels(G, pathpair_list)
                        #self.logger.debug(f"rewards: {rewards}")
                        return 1
                except Exception as e:
                    #self.logger.error(str(e)+"occurred, try next pair monitor")
                    return 2
            else:
                return 2

        else:
            return 3


    def construct_pathPair_from_path(self, path):
        pathpair = []
        for i in range(len(path) - 1):
            if (path[i], path[i + 1]) in self.Dict_edge_theta:
                pathpair.append((path[i], path[i + 1]))
            else:
                pathpair.append((path[i + 1], path[i]))
        return pathpair

    def update_MBA_variabels(self, G, pathpair):
        for edge in pathpair:
            # print(f"edge {edge}")
            ##to check
            self.Dict_edge_theta[edge] = (self.Dict_edge_theta[edge]*self.Dict_edge_m[edge] + G[edge[0]][edge[1]]['delay']) / (
                    self.Dict_edge_m[edge] + 1)
            self.Dict_edge_m[edge] = self.Dict_edge_m[edge] + 1
            # print(f"G[edge[0]][edge[1]]: {G[edge[0]][edge[1]]}")
            #total_weight += G[edge[0]][edge[1]]['weight']
            #G[edge[0]][edge[1]]['weight'] += G[edge[0]][edge[1]]['weight']+5
            G.edges.data()
            #print("the edge is selected in the shortest path according to the parameter ")
        #self.logger.debug(f"Dict_edge_m: {self.Dict_edge_m}")
        #self.logger.debug(f"Dict_edge_theta[edge]: {self.Dict_edge_theta}")
        #print(f"rewards: {rewards}")

    def optimal_path(self, G, source, destination):
        optimal_delay = 0
        optimal_path = nx.shortest_path(G, source=source, target=destination, weight='delay_mean', method='dijkstra')
        self.logger.info(f"optimal path: {optimal_path}")
        for i in range(len(optimal_path) - 1):
            optimal_delay += G[optimal_path[i]][optimal_path[i + 1]]["delay_mean"]
        self.logger.info(f"optimal_delay: {optimal_delay}")
        return optimal_delay, optimal_path

    def LLC_policy_explore(self, G, source, destination, total_rewards, offset):
        # select a path which solves the minimization problem
        for edge in G.edges:
            # llc_factor=Dict_edge_theta[edge] + math.sqrt((len(G.edges) + 1) * math.log(t) / Dict_edge_m[edge])
            # print(f"llc_factor: {llc_factor}")

            #G[edge[0]][edge[1]]["llc_factor"] = self.Dict_edge_theta[edge] + math.sqrt(self.Dict_edge_m[edge]/(len(G.edges) + 1) * math.log(self.t))
            G[edge[0]][edge[1]]["llc_factor"] = self.Dict_edge_theta[edge] - math.sqrt((len(G.edges) + 1) * math.log(self.t) /self.Dict_edge_m[edge])+ self.Dict_edge_m[edge]* offset
            self.logger.debug(f"minus sqrt factor{edge}:{math.sqrt((len(G.edges) + 1) * math.log(self.t) / self.Dict_edge_m[edge])}")
            '''
            G[edge[0]][edge[1]]["llc_factor"] = self.Dict_edge_theta[edge] - math.sqrt(
                (len(G.edges) + 1) * math.log(self.t) / self.Dict_edge_m[edge]) + offset
            '''
        # select the shortest path with wrt the llc_fact
        #print(G.edges.data())
        shortest_path = nx.shortest_path(G, source=source, target=destination, weight='llc_factor', method='dijkstra')
        self.logger.info(f"shortest path: {shortest_path}")
        # print(G.edges.data())
        # update the Dict_edge_theta and Dict_edge_m
        pathpair= self.construct_pathPair_from_path(shortest_path)
        rewards=self.update_MBA_variabels(G,pathpair)
        self.logger.debug(f"rewards:{rewards}")
        total_rewards.append(rewards)
        # print(f"total_rewards:{total_rewards}")
        #print(Dict_edge_theta)
        return total_rewards, shortest_path

    def LLC_policy_exploitation(self, G, monitor1, monitor2):
        # select a path which solves the minimization problem
        #Todo modify the LLC policy exploitaiton with multiple monitors

        for edge in G.edges:
            # llc_factor=Dict_edge_theta[edge] + math.sqrt((len(G.edges) + 1) * math.log(t) / Dict_edge_m[edge])
            # print(f"llc_factor: {llc_factor}")
            # G[edge[0]][edge[1]]["llc_factor"] = self.Dict_edge_theta[edge] + math.sqrt(self.Dict_edge_m[edge]/(len(G.edges) + 1) * math.log(self.t)
            llc_factor= self.Dict_edge_theta[edge] - 0.6*math.sqrt(
                (len(G.edges) + 1) * math.log(self.t) / self.Dict_edge_m[edge])
            if llc_factor < 0:
                G[edge[0]][edge[1]]["llc_factor"]=0
                #print(f"edge{edge[0]} {edge[1]} got negtive lcc factor:")
                #print({G[edge[0]][edge[1]]["llc_factor"]})
            else:
                G[edge[0]][edge[1]]["llc_factor"] = llc_factor
            #self.logger.debug(
            #    f"minus sqrt factor{edge}:{math.sqrt((len(G.edges) + 1) * math.log(self.t) / self.Dict_edge_m[edge])}")
            '''
            G[edge[0]][edge[1]]["llc_factor"] = self.Dict_edge_theta[edge] - math.sqrt(
                (len(G.edges) + 1) * math.log(self.t) / self.Dict_edge_m[edge]) + offset
            '''

        # select the shortest path with wrt the llc_fact
        # print(G.edges.data())
        shortest_path = nx.shortest_path(G, source=monitor1, target=monitor2, weight='llc_factor', method='dijkstra')
        #print(f"t={self.t} shortest path {shortest_path}")
        #self.logger.info(f"shortest path: {shortest_path}")
        # print(G.edges.data())
        # update the Dict_edge_theta and Dict_edge_m
        pathpair = self.construct_pathPair_from_path(shortest_path)
        self.update_MBA_variabels(G, pathpair)
        #self.logger.debug(f"rewards:{rewards}")
        #total_rewards.append(rewards)
        # print(f"total_rewards:{total_rewards}")
        # print(Dict_edge_theta)
        #return total_rewards, shortest_path

    def train_llc(self,G, time, monitor_pair_list,source, destination):
        #offset = math.sqrt((len(G.edges) + 1) * math.log(time)) #shift the llc factor > 0
        #self.logger.debug(f"offset: {offset}")
        optimal_path=nx.shortest_path(G,source, destination, 'delay-mean')
        optimal_delay=nx.path_weight(G,optimal_path,'delay')
        selected_shortest_path=[]
        #print(f"minimum lcc {math.sqrt((len(G.edges) + 1) * math.log(t + round)/self)}")
        mse=[]
        total_mse_array = []
        total_rewards = []   #in the current implementation, it is for only one pair of monitors
        total_regrets = []
        self.logger.debug(f"t={self.t}, start trainning...")
        for i in range(time-self.t):
            #self.logger.info(f"t= {self.t}")
            # i<3000:
            for monitor_pair in monitor_pair_list:
                m1=monitor_pair[0]
                m2=monitor_pair[1]
                self.LLC_policy_exploitation(G, m1, m2)

            #else:
            #    total_rewards,shortest_path=self.LLC_policy_exploitation(G, source, destination, total_rewards, offset)
            shortest_path=nx.shortest_path(G, source=source, target=destination, weight='llc_factor', method='dijkstra')
            selected_shortest_path.append(shortest_path)
            #self.logger.debug(f"t={self.t}, selected shortest path:{selected_shortest_path}")
            rewards=nx.path_weight(G,shortest_path,'delay')
            total_rewards.append(rewards)
            regret = sum(total_rewards) - self.t * optimal_delay
            total_regrets.append(regret)
            #self.logger.info(f"regretes:{regret}")
           # print(total_regrets)
            total_mse = 0
            for edge in G.edges:
                total_mse+=(self.Dict_edge_theta[edge]-G[edge[0]][edge[1]]['delay_mean'])**2
            total_mse_array.append(total_mse/len(G.edges))

            '''for debugging the llc factor problem
            if self.t==327:
                print("t=327")
                print("path (3,45,47)")
                print(f"m(3,45)={self.Dict_edge_m[('3', '45')]}, theta={self.Dict_edge_theta[('3','45')]}, llc={G['3']['45']['llc_factor']}")
                print(f"m(45,47)={self.Dict_edge_m[('45', '47')]}, theta={self.Dict_edge_theta[('45', '47')]}, llc={G['45']['47']['llc_factor']}")
                print(f"path weight = {nx.path_weight(G,['3', '45', '47'],'llc_factor')}")

                print("path (3,49,30,40,47)")
                print(f"m(3,49)={self.Dict_edge_m[('3', '49')]}, theta={self.Dict_edge_theta[('3', '49')]}, llc={G['3']['49']['llc_factor']}")
                print(f"m(30,49)={self.Dict_edge_m[('30', '49')]}, theta={self.Dict_edge_theta[('30', '49')]}, llc={G['30']['49']['llc_factor']}")
                print(f"m(30,40)={self.Dict_edge_m[('30', '40')]}, theta={self.Dict_edge_theta[('30', '40')]}, llc={G['30']['40']['llc_factor']}")
                print(f"m(40,47)={self.Dict_edge_m[('40', '47')]}, theta={self.Dict_edge_theta[('40', '47')]}, llc={G['40']['47']['llc_factor']}")
                print(f"path weight = {nx.path_weight(G, ['3', '49', '30', '40', '47'], 'llc_factor')}")



            if self.t==3000:
                print('t=3000')
                print("path (3,45,47)")
                print(
                    f"m(3,35)={self.Dict_edge_m[('3', '45')]}, theta={self.Dict_edge_theta[('3', '45')]}, llc={G['3']['45']['llc_factor']}")
                print(
                    f"m(45,47)={self.Dict_edge_m[('45', '47')]}, theta={self.Dict_edge_theta[('45', '47')]}, llc={G['45']['47']['llc_factor']}")
                print(f"path weight = {nx.path_weight(G, ['3', '45', '47'], 'llc_factor')}")

                print("path (3,49,30,40,47)")
                print(
                    f"m(3,49)={self.Dict_edge_m[('3', '49')]}, theta={self.Dict_edge_theta[('3', '49')]}, llc={G['3']['49']['llc_factor']}")
                print(
                    f"m(30,49)={self.Dict_edge_m[('30', '49')]}, theta={self.Dict_edge_theta[('30', '49')]}, llc={G['30']['49']['llc_factor']}")
                print(
                    f"m(30,40)={self.Dict_edge_m[('30', '40')]}, theta={self.Dict_edge_theta[('30', '40')]}, llc={G['30']['40']['llc_factor']}")
                print(
                    f"m(40,47)={self.Dict_edge_m[('40', '47')]}, theta={self.Dict_edge_theta[('40', '47')]}, llc={G['40']['47']['llc_factor']}")
                print(f"path weight = {nx.path_weight(G, ['3', '49', '30', '40', '47'], 'llc_factor')}")
            '''
            self.t = self.t + 1  # the time slot increase 1
            #for plotting
            #self.logger.debug(f"t={self.t}, trainning...")
            self.topo.assign_link_delay(G)

            if self.t==1000:
                delay_difference1 = []
                for edge in G.edges:
                    delay_difference1.append(abs(self.Dict_edge_theta[edge]-G[edge[0]][edge[1]]['delay_mean']))
                self.edge_delay_difference_list.append(delay_difference1)
                self.plotter.plot_edge_delay_difference(G,self.Dict_edge_theta)
                plt.savefig(self.directory + 'delay difference from mean at t=1000', format="PNG", dpi=300,
                            bbox_inches='tight')
            if self.t==2000:
                delay_difference2 = []
                for edge in G.edges:
                    delay_difference2.append(abs(self.Dict_edge_theta[edge] - G[edge[0]][edge[1]]['delay_mean']))
                self.edge_delay_difference_list.append(delay_difference2)
                self.plotter.plot_edge_delay_difference(G,self.Dict_edge_theta)
                plt.savefig(self.directory + 'delay difference from mean at t=2000', format="PNG", dpi=300,
                            bbox_inches='tight')
            if self.t==3000:
                delay_difference3 = []
                for edge in G.edges:
                    delay_difference3.append(abs(self.Dict_edge_theta[edge] - G[edge[0]][edge[1]]['delay_mean']))
                self.edge_delay_difference_list.append(delay_difference3)
                self.plotter.plot_edge_delay_difference(G,self.Dict_edge_theta)
                plt.savefig(self.directory + 'delay difference from mean at t=3000', format="PNG", dpi=300,
                            bbox_inches='tight')

        self.t=1
        self.plotter.plot_total_edge_delay_mse(total_mse_array)
        self.plotter.plot_total_rewards(total_rewards,optimal_delay)
        #plot the delay difference from the mean along time
        self.plotter.plot_edge_delay_difference_alongtime(0,15,self.edge_delay_difference_list)
        plt.savefig(self.directory + 'delay difference from the mean like 0-14', format="PNG")
        self.plotter.plot_edge_delay_difference_alongtime(15, 30,self.edge_delay_difference_list)
        plt.savefig(self.directory + 'delay difference from the mean like 15-30', format="PNG")
        self.plotter.plot_edge_delay_difference_alongtime(30, 42,self.edge_delay_difference_list)
        plt.savefig(self.directory + 'delay difference from the mean like 31-42', format="PNG")
        #plot the number of edges has been explored after the training

        self.plotter.plot_edge_exploitation_times_bar('t=3000',self.Dict_edge_m)
        plt.savefig(self.directory + '# of edge exploration after training', format="PNG", dpi=300,
                    bbox_inches='tight')
        self.edge_exploration_times.append([v for k, v in self.Dict_edge_m.items()])
        self.plotter.plot_edge_exploitation_times_bar_combined(self.edge_exploration_times)

        #check how many edges has been explored during the training
        self.logger.debug("training is finished")
        self.logger.debug(f"Dict_edge_m values are added to the edge_exploration_times array")
        self.logger.debug(f"{self.edge_exploration_times}")
        init=np.array(self.edge_exploration_times[0])
        end=np.array(self.edge_exploration_times[1])

        expo_count=0
        for i in range (len(G.edges)):
            if end[i] > init[i]:
                expo_count+=1
        self.edge_exploration_times=[]
        return total_rewards,selected_shortest_path, expo_count, total_mse_array














