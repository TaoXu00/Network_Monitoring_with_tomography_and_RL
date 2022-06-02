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

class multi_armed_bandit:
    # main(100, 0.5)
    def __init__(self, topo):
        self.topo=topo
        self.Dict_edge_theta = {}
        self.Dict_edge_m = {}
        self.t=1
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

        total_rewards = []   #don't know what to do with these three variables
        total_regrets = []
        sample_delays = []

        monitor_pair_list=combinations(monitors, 2)
        print("initializing..........")
        for edge in G.edges:   #traversal each link to guarantee every link is covered at least once
            n1 = edge[0]
            n2 = edge[1]
            for monitor_pair in monitor_pair_list:
                print(monitor_pair[0], monitor_pair[1])
                source = monitor_pair[0]
                destination = monitor_pair[1]
                optimal_delay = self.optimal_path(G, source, destination)   #don't know why we need to find the optimal delay
                # find the shortest path between source to n1 and n2 to destination
                '''store the sampled delay for each link'''
                delays = []
                for edge in G.edges:
                    delays.append(G[edge[0]][edge[1]]['delay'])
                # print(delays)
                sample_delays.append(delays)
                # index of time slot
                shortest_path_l = nx.shortest_path(G, source=source, target=n1, weight='weight',
                                                 method='dijkstra')
                print(f" t = {self.t}, shortest path from {source} to {n1}: {shortest_path_l}")
                shortest_path_r = nx.shortest_path(G, source=n1, target=destination, weight='weight',
                                                 method='dijkstra')
                print(f" t = {self.t}, shortest path from {source} to {n1}: {shortest_path_l}")
                if(len(shortest_path_r)!=0 and len(shortest_path_r)!=0):
                    pathpair_list_l=self.construct_pathPair_from_path(shortest_path_l)
                    pathpair_list_r=self.construct_pathPair_from_path(shortest_path_r)
                    pathpair_list=pathpair_list_l+pathpair_list_r
                    self.update_MBA_variabels(pathpair_list)
                else:
                    break


                # observe the links in the returned shortest path and update the theta vector and m vector
                total_weight = 0

                # if shortest_path not in shortest_path_list:
                #    print("Detect a new shortest path")
                #    rewards=0
                #    shortest_path_list.append(shortest_path)
                rewards = 0

                total_rewards.append(rewards)
                # print(f"total_rewards: {total_rewards}")
                regret = sum(total_rewards) - self.t * optimal_delay
                total_regrets.append(regret)
                # print(f"regret: {regret}")
                # print(f"total_regrets: {total_regrets}")
                # print(Dict_edge_theta)
                # print(Dict_edge_m)
                self.topo.assign_link_delay(G, self.topo.Dict_edge_scales)
                self.t = self.t + 1
                counter = 0
                for item in self.Dict_edge_m.values():
                    if item != 0:
                        counter = counter + 1


            '''
            #old method.....
            num = len(list(nx.all_simple_paths(G, source=source, target=destination)))
            print(f"the total path number is {num}")
            # maintain t,wo vectors (1*N(#of edge)), theta and m. theta is the average(sample mean) of all the observed values of Xi up to the
            # current time-slot, m is the number of times that Xi has been observed up to the current time-slot.
            Dict_edge_theta = {}
            Dict_edge_m = {}
            for edge in G.edges:
                Dict_edge_theta[edge] = 0
            for edge in G.edges:
                Dict_edge_m[edge] = 0

            counter = 0
            t = 1
            total_rewards = []
            total_regrets = []
            sample_delays = []
            print("initializing..........")
            for edge in G.edges:
                n1=edge[0]
                n2=edge[1]



            while (counter != len(G.edges)):
                #store the sampled delay for each link
                delays = []
                for edge in G.edges:
                    delays.append(G[edge[0]][edge[1]]['delay'])
                # print(delays)
                sample_delays.append(delays)
                # index of time slot
                shortest_path = nx.shortest_path(G, source=source, target=destination, weight='weight', method='dijkstra')
                print(f" t = {t}, shortest path: {shortest_path}")
                # observe the links in the returned shortest path and update the theta vector and m vector
                total_weight = 0
                pathpair = []
                for i in range(len(shortest_path) - 1):
                    if (shortest_path[i], shortest_path[i + 1]) in Dict_edge_theta:
                        pathpair.append((shortest_path[i], shortest_path[i + 1]))
                    else:
                        pathpair.append((shortest_path[i + 1], shortest_path[i]))
                # if shortest_path not in shortest_path_list:
                #    print("Detect a new shortest path")
                #    rewards=0
                #    shortest_path_list.append(shortest_path)
                rewards = 0
                for edge in pathpair:
                    #print(f"edge {edge}")
                    Dict_edge_theta[edge] = (Dict_edge_theta[edge] + G[edge[0]][edge[1]]['delay']) / (Dict_edge_m[edge] + 1)
                    Dict_edge_m[edge] = Dict_edge_m[edge] + 1
                    #print(f"G[edge[0]][edge[1]]: {G[edge[0]][edge[1]]}")
                    total_weight += G[edge[0]][edge[1]]['weight']
                    G[edge[0]][edge[1]]['weight'] += random.randint(1, 5)
                    G.edges.data()
                    rewards += G[edge[0]][edge[1]]['delay']
                #print(f"Dict_edge_m: {Dict_edge_m}")
                #print(f"Dict_edge_theta[edge]: {Dict_edge_theta}")
                #print(f"rewards: {rewards}")
                total_rewards.append(rewards)
                #print(f"total_rewards: {total_rewards}")
                regret = sum(total_rewards) - t * optimal_delay
                total_regrets.append(regret)
                # print(f"regret: {regret}")
                # print(f"total_regrets: {total_regrets}")
                # print(Dict_edge_theta)
                # print(Dict_edge_m)
                self.topo.assign_link_delay(G, Dict_edge_scales)
                t = t + 1
                counter = 0
                for item in Dict_edge_m.values():
                    if item != 0:
                        counter = counter + 1
        print(
            "===============================Initialization is finished======================================================== ")
        return Dict_edge_theta, Dict_edge_m, t, total_rewards, total_regrets
        '''

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
            self.Dict_edge_theta[edge] = (self.Dict_edge_theta[edge] + G[edge[0]][edge[1]]['delay']) / (
                    self.Dict_edge_m[edge] + 1)
            self.Dict_edge_m[edge] = self.Dict_edge_m[edge] + 1
            # print(f"G[edge[0]][edge[1]]: {G[edge[0]][edge[1]]}")
            #total_weight += G[edge[0]][edge[1]]['weight']
            G[edge[0]][edge[1]]['weight'] += G[edge[0]][edge[1]]['weight']+5
            G.edges.data()
            #rewards += G[edge[0]][edge[1]]['delay']
        # print(f"Dict_edge_m: {Dict_edge_m}")
        # print(f"Dict_edge_theta[edge]: {Dict_edge_theta}")
        # print(f"rewards: {rewards}")

    def optimal_path(self, G, source, destination):
        optimal_delay = 0
        shortest_path = nx.shortest_path(G, source=source, target=destination, weight='delay-mean', method='dijkstra')
        print(f"optimal path: {shortest_path}")
        for i in range(len(shortest_path) - 1):
            optimal_delay += G[shortest_path[i]][shortest_path[i + 1]]["delay-mean"]
        print(f"optimal_delay: {optimal_delay}")
        return optimal_delay

    def LLC_policy(self, G, Dict_edge_theta, Dict_edge_m, t, source, destination, total_rewards, offset):
        # select a path which solves the minimization problem
        for edge in G.edges:
            # llc_factor=Dict_edge_theta[edge] + math.sqrt((len(G.edges) + 1) * math.log(t) / Dict_edge_m[edge])
            # print(f"llc_factor: {llc_factor}")
            G[edge[0]][edge[1]]["llc_factor"] = Dict_edge_theta[edge] - math.sqrt(
                (len(G.edges) + 1) * math.log(t) / Dict_edge_m[edge]) + Dict_edge_m[edge]*offset
        # select the shortest path with wrt the llc_fact
        #print(G.edges.data())
        shortest_path = nx.shortest_path(G, source=source, target=destination, weight='llc_factor', method='dijkstra')
        print(f"shortest path: {shortest_path}")
        # print(G.edges.data())
        # update the Dict_edge_theta and Dict_edge_m
        pathpair = []
        rewards = 0
        for i in range(len(shortest_path) - 1):
            if (shortest_path[i], shortest_path[i + 1]) in Dict_edge_theta:
                pathpair.append((shortest_path[i], shortest_path[i + 1]))
            else:
                pathpair.append((shortest_path[i + 1], shortest_path[i]))
        #print(f"pathpair{pathpair}")
        #print(f"G.edges {G.edges}")

        for edge in pathpair:
            #print(f"edge {edge}")
            #print(
            #    f"Dict_edge_theta[edge] {Dict_edge_theta[edge]} +  G[edge[0]][edge[1]]['delay']: {G[edge[0]][edge[1]]['delay']}")
            #print(f"Dict_edge_m[edge] {Dict_edge_m[edge]}")
            Dict_edge_theta[edge] = (Dict_edge_theta[edge] + G[edge[0]][edge[1]]['delay']) / (Dict_edge_m[edge] + 1)
            Dict_edge_m[edge] = Dict_edge_m[edge] + 1
            rewards = rewards + G[edge[0]][edge[1]]['delay']
        print(f"Dict_edge_m: {Dict_edge_m}")
        print(f"Dict_edge_theta: {Dict_edge_theta}")
        print(f"rewards:{rewards}")
        total_rewards.append(rewards)
        # print(f"total_rewards:{total_rewards}")
        #print(Dict_edge_theta)
        return total_rewards

    def train_llc(self, G, source, destination, round, Dict_edge_theta, Dict_edge_m, optimal_delay, total_rewards, t,
                  Dict_edge_scales, total_regrets):
        offset = math.sqrt((len(G.edges) + 1) * math.log(t + round))
        #print(f"offset: {offset}")
        #print(f"minimum lcc {math.sqrt((len(G.edges) + 1) * math.log(t + round)/self)}")

        for i in range(round):
            print(f"t={t}")
            self.topo.assign_link_delay(G, Dict_edge_scales)
            total_rewards = self.LLC_policy(G, Dict_edge_theta, Dict_edge_m, t, source, destination, total_rewards, offset)
            regret = sum(total_rewards) - t * optimal_delay
            print(f"regretes:{regret}")
            total_regrets.append(regret)
           # print(total_regrets)
            t = t + 1  # the time slot increase 1
        return total_regrets, t

