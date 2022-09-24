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
    def __init__(self, topo, logger, directory, nt):
        self.topo=topo
        self.plotter=plotter.plotter(directory)
        self.Dict_edge_theta = {}   #the observed actual average delay-mean for each edge
        self.Dict_edge_m = {}       #the counter for how many times this edge has been observed
        self.t=1
        self.logger=logger
        self.directory=directory
        self.edge_delay_difference_list=[]
        self.edge_exploration_times=[]
        self.nt=nt

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
            #self.logger.debug("t= %s initializing edge %s, %s" %(self.t, n1, n2))
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
                if found == 1:
                    #self.logger.debug("found the path with monitor1 %s monitor2 %s and left_node %s right_node %s" %(m1, m2, n1, n2))
                    break
                elif found ==2:
                    #self.logger.debug("return 2 in Check1")
                    continue
                elif found==3 or found==4:   #the right node is the same as the source node, now check the edge (n2, n1)
                    #self.logger.debug("return 3 in check 1, found = %d" %(found))
                    found_inverse=self.find_path(G, edge_G, n2, n1, m1, m2)
                    if found_inverse==2:
                        #self.logger.debug("return 2 in check 2")
                        continue
                    elif found_inverse==1:
                        #self.logger.debug("found the path with monitor %s monitor %s and left_node %s right_node %s" %(m1,m2,n2,n1))
                        break
                    elif found_inverse==4 or found_inverse==3:
                        #self.logger.debug(f"check with next monitor pair")
                        continue




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
        #self.logger.debug("Dict_edge_m: %s" %(self.Dict_edge_m))
        #self.logger.debug(f"Dict_edge_theta[edge]: {self.Dict_edge_theta}")
        self.plotter.plot_edge_delay_difference(G, self.Dict_edge_theta)
        delay_difference=[]
        for edge in G.edges:
            delay_difference.append(abs(self.Dict_edge_theta[edge]-G[edge[0]][edge[1]]['delay_mean']))
        self.edge_delay_difference_list.append(delay_difference)
        plt.savefig(self.directory + 'delay difference from mean after initialization', format="PNG", dpi=300, bbox_inches='tight')
        self.plotter.plot_edge_exploitation_times_bar('After initialization',self.Dict_edge_m)
        self.edge_exploration_times.append([v for k, v in self.Dict_edge_m.items() ])
        #self.logger.debug("Dict_edge_m values are added to the edge_exploration_times array")
        #self.logger.debug("%s" %(self.edge_exploration_times))
        plt.savefig(self.directory + '# of edge exploration after initialization ', format="PNG", dpi=300,
                    bbox_inches='tight')


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
                #self.logger.debug("shortest path from %s to %s: %s" %(source, left_node,shortest_path_l))
                pathpair_list_l = self.construct_pathPair_from_path(shortest_path_l)
                G_r = G.copy()
                # print(G_r.nodes)
                for edge in pathpair_list_l:
                    if edge in G_r.edges:
                        G_r.remove_edge(edge[0], edge[1])
                    else:
                        G_r.remove_edge(edge[1], edge[0])
            except Exception as e:
                self.logger.error(str(e)+"occurred, try the inversed direction.")
                return 3
            G_r.remove_node(left_node)
            if(source in G_r.nodes):
                G_r.remove_node(source)
            if (destination in G_r.nodes and right_node in G_r.nodes):
                try:
                    shortest_path_r = nx.shortest_path(G_r, source=right_node, target=destination, weight='weight',
                                               method='dijkstra')
                    #.logger.debug("shortest path from %s to %s: %s" %(right_node,destination,shortest_path_r))
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
                    #self.logger.error(str(e) + "occurred, try the inversed direction.")
                    return 4
                    #self.logger.error(str(e) + "occurred, try the inversed direction.")
                    #return 3
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

    def update_MBA_variabels(self, G, pathpair):   ##different
        for e in pathpair:
            if e not in list(G.edges):
                edge = (e[1], e[0])
            else:
                edge = e
            self.Dict_edge_theta[edge] = (self.Dict_edge_theta[edge] * self.Dict_edge_m[edge] + G[edge[0]][edge[1]][
                'delay']) / (self.Dict_edge_m[edge] + 1)
            self.Dict_edge_m[edge] = self.Dict_edge_m[edge] + 1

    def update_MBA_variabels_with_NT(self, G, x, explored_edge_set, edge_average_delay_Dict):
        edges = list(G.edges)
        for i in range(len(x)):
            if x[i] != 0:
                edge = edges[i]
                #self.logger.debug(
                #    "edge %s is computed with the value %f, its realtime delay is %f " %(edge, x[i],G[edge[0]][edge[1]]['delay']))
                self.Dict_edge_theta[edge] = (self.Dict_edge_theta[edge] * self.Dict_edge_m[edge] + x[i]) / (
                        self.Dict_edge_m[edge] + 1)
                self.Dict_edge_m[edge] = self.Dict_edge_m[edge] + 1
                if edge not in explored_edge_set:
                    edge = (edge[1], edge[0])
                explored_edge_set.remove(edge)
        for edge in explored_edge_set:
            if edge not in edges:
                edge_g = (edge[1], edge[0])
            else:
                edge_g=edge
            #as a base line approach, the unindentificable links will be assigned with average delay(total_path_delay/path_length)
            self.Dict_edge_theta[edge_g] = (self.Dict_edge_theta[edge_g] * self.Dict_edge_m[edge_g] + edge_average_delay_Dict[edge]) / (
                    self.Dict_edge_m[edge_g] + 1)
            self.Dict_edge_m[edge_g] = self.Dict_edge_m[edge_g] + 1
            #self.logger.debug("updating %s with average %s" %(edge, edge_average_delay_Dict[edge]))

    def optimal_path(self, G, monitor_pair_list):
        optimal_delay_dict = {}
        optimal_path_dict={}
        for monitor_pair in monitor_pair_list:
            optimal_path = nx.shortest_path(G, source=monitor_pair[0], target=monitor_pair[1], weight='delay_mean', method='dijkstra')
            #self.logger.info("optimal path: %s" %(optimal_path))
            optimal_delay=0
            for i in range(len(optimal_path) - 1):
                optimal_delay += G[optimal_path[i]][optimal_path[i + 1]]["delay_mean"]
            optimal_delay_dict[monitor_pair]=optimal_delay
            optimal_path_dict[monitor_pair]=optimal_path
        #self.logger.info("optimal_delay: %s" %(optimal_delay_dict))
        return optimal_delay_dict,optimal_path_dict


    def LLC_policy(self, G, monitor1, monitor2, llc):
        # select a path which solves the minimization problem
        print(llc)
        for edge in G.edges:
            llc_factor= self.Dict_edge_theta[edge] - llc * math.sqrt(
                (len(G.edges) + 1) * math.log(self.t) / self.Dict_edge_m[edge])
            if llc_factor < 0:
                G[edge[0]][edge[1]]["llc_factor"]=0
                print("edge (%s %s) got negtive lcc factor" %(edge[0],edge[1]))
            else:
                G[edge[0]][edge[1]]["llc_factor"] = llc_factor

        shortest_path = nx.shortest_path(G, source=monitor1, target=monitor2, weight='llc_factor', method='dijkstra')
        return shortest_path
    def LLC_policy_without_MAB(self, G, monitor1, monitor2):
        # select a path which solves the minimization problem
        for edge in G.edges:
            average= self.Dict_edge_theta[edge]
            G[edge[0]][edge[1]]["average-delay"]=average
        shortest_path = nx.shortest_path(G, source=monitor1, target=monitor2, weight='average-delay', method='dijkstra')
        return shortest_path

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

    def train_llc(self,G, time, monitor_pair_list, llc_factor):
        optimal_delay_dict, optimal_path_dict = self.optimal_path(G, monitor_pair_list)
        optimal_links=[]
        for key in optimal_path_dict:
            path = optimal_path_dict[key]
            path_pair = self.construct_pathPair_from_path(path)
            for pair in path_pair:
                if pair not in optimal_links:
                    optimal_links.append(pair)
        selected_shortest_path=[]
        total_mse_array = []
        total_rewards_dict = {}   #in the current implementation, it is for only one pair of monitors
        computed_edge_num=[]
        correct_shortest_path_selected_rate = []
        optimal_edges_delay_difference_after_inti=[]
        optimal_edges_delay_difference_after_training=[]

        for link in optimal_links:
            optimal_edges_delay_difference_after_inti.append(abs(self.Dict_edge_theta[link] - G[link[0]][link[1]]["delay_mean"]))

        self.logger.debug("start trainning...")
        for monitor_pair in monitor_pair_list:
            total_rewards_dict[monitor_pair]=[]
        diff_of_delay_from_optimal_real_time=[]
        Dict_time_of_optimal_path_selected = {}
        rate_optimal_path_selected = []
        T_total=time-self.t
        for monitor_pair in monitor_pair_list:
            Dict_time_of_optimal_path_selected[monitor_pair] = []
        for i in range(T_total):
            #self.logger.info("t= %s" %(self.t))
            total_mse = 0
            for edge in G.edges:
                total_mse += (self.Dict_edge_theta[edge] - G[edge[0]][edge[1]]['delay_mean']) ** 2
            total_mse_array.append(total_mse / len(G.edges))
            explored_path_list = []
            num_correct_shortest_path = 0
            total_diff=0
            for monitor_pair in monitor_pair_list:
                m1=monitor_pair[0]
                m2=monitor_pair[1]
                shortest_path=self.LLC_policy(G, m1, m2, llc_factor)
                #shortest_path = self.LLC_policy_without_MAB(G, m1, m2)
                if shortest_path == optimal_path_dict[monitor_pair] and self.t>=T_total-1001:
                    num_correct_shortest_path +=1
                    Dict_time_of_optimal_path_selected[monitor_pair].append(1)
                else:
                    Dict_time_of_optimal_path_selected[monitor_pair].append(0)
                #else:  #check how far it is different from the real optimal path
                    #total_diff+= abs(nx.path_weight(G,shortest_path,"delay_mean")- optimal_delay_dict[monitor_pair])
                total_diff += abs(nx.path_weight(G,optimal_path_dict[monitor_pair], "delay")-nx.path_weight(G, shortest_path, "delay"))
                explored_path_list.append(shortest_path)
                rewards = nx.path_weight(G, shortest_path, 'delay')
                total_rewards_dict[monitor_pair].append(rewards)
            diff_of_delay_from_optimal_real_time.append(total_diff/len(optimal_delay_dict))
            path_list = []
            for path in explored_path_list:
                pathpair = []
                for i in range(len(path) - 1):
                    pathpair.append((path[i], path[i + 1]))
                path_list.append(pathpair)
            #self.logger.debug("path_list: %s" %(path_list))
            edge_average_delay_list_dict={}
            edge_average_delay_dict={}
            b, average_delay_list = self.end_to_end_measurement(G, path_list)
            for i in range(len(average_delay_list)):
                for edge in path_list[i]:
                    if edge not in edge_average_delay_list_dict:
                        edge_average_delay_list_dict[edge] = []
                    edge_average_delay_list_dict[edge].append(average_delay_list[i])
            for edge in edge_average_delay_list_dict:
                edge_average_delay_dict[edge]=sum(edge_average_delay_list_dict[edge])/len(edge_average_delay_list_dict[edge])
            #self.logger.debug("Dict=%s" %(edge_average_delay_dict))
            # get the explored edge set
            explored_edge_set = []
            for path in path_list:
                for edge in path:
                    if (edge[0], edge[1]) not in explored_edge_set and (edge[1], edge[0]) not in explored_edge_set:
                        explored_edge_set.append(edge)
            #call NT as a submoudle
            x, count = self.nt.nt_engine(G, path_list, b)
            #self.logger.debug("%d edges has been explored, they are: %s" %(len(explored_edge_set),explored_edge_set))
            #print("x=%s %d edges are computed" %(x, count))
            computed_edge_num.append(count)
            correct_shortest_path_selected_rate.append(num_correct_shortest_path / len(monitor_pair_list))
            # the MBA variables should be updated according to the results computed by the NT.
            self.update_MBA_variabels_with_NT(G, x, explored_edge_set, edge_average_delay_dict)
            #self.update_MBA_variabels(G, explored_edge_set)


            #shortest_path=nx.shortest_path(G, source=source, target=destination, weight='llc_factor', method='dijkstra')
            #selected_shortest_path.append(shortest_path)
            #self.logger.debug(f"t={self.t}, selected shortest path:{selected_shortest_path}")
            '''
            total_rewards.append(rewards)
            regret = sum(total_rewards) - self.t * optimal_delay
            total_regrets.append(regret)
            '''
            self.t = self.t + 1  # the time slot increase 1
            self.topo.assign_link_delay(G)
            self.plot_edge_delay_difference_at_different_time_point(G)
        for monitor_pair in monitor_pair_list:
            count_list = Dict_time_of_optimal_path_selected[monitor_pair]
            rate = sum(count_list[-1000:])/1000
            rate_optimal_path_selected.append(rate)
        average_optimal_path_selected_rate = np.average(np.array(rate_optimal_path_selected))
        correct_shortest_path_selected_rate_array=np.array(correct_shortest_path_selected_rate)
        average_optimal_path_selected_rate_among_monitor_pairs= np.average(np.array(correct_shortest_path_selected_rate_array[-1000:]))
        avg_diff_of_delay_from_optimal = (sum(diff_of_delay_from_optimal_real_time) / len(diff_of_delay_from_optimal_real_time))
        rewards_mse_list=self.compute_rewards_mse(total_rewards_dict, optimal_delay_dict)
        average_regret_list = self.compute_regret(total_rewards_dict, optimal_delay_dict)
        self.plotter.plot_total_edge_delay_mse(total_mse_array)
        self.plotter.plot_time_average_rewards(rewards_mse_list)
        self.plotter.plot_average_regrets(average_regret_list)
        self.plotter.plot_diff_from_optimal_path_of_selected_shortest_paths(diff_of_delay_from_optimal_real_time)
        #plot the delay difference from the mean along time
        self.plotter.plot_edge_delay_difference_alongtime(0,15,self.edge_delay_difference_list,'0-15')
        self.plotter.plot_edge_delay_difference_alongtime(15, 30,self.edge_delay_difference_list,'15-30')
        self.plotter.plot_edge_delay_difference_alongtime(30, 35,self.edge_delay_difference_list, '30-35')

        self.plotter.plot_edge_exploitation_times_bar('t=3000',self.Dict_edge_m)
        plt.savefig(self.directory + '# of edge exploration after training', format="PNG", dpi=300,
                    bbox_inches='tight')
        self.edge_exploration_times.append([v for k, v in self.Dict_edge_m.items()])
        self.plotter.plot_edge_exploitation_times_bar_combined(self.edge_exploration_times)
        self.plotter.plot_rate_of_correct_shortest_path(correct_shortest_path_selected_rate)  # implement this function


        #check how many edges has been explored during the training
        self.logger.debug("training is finished")
        #self.logger.debug("Dict_edge_m values are added to the edge_exploration_times array")
        self.logger.debug("%s" %(self.edge_exploration_times))
        init=np.array(self.edge_exploration_times[0])
        end=np.array(self.edge_exploration_times[1])
        expo_count=0
        edge_exploration_during_training = []
        for i in range (len(G.edges)):
            if end[i] > init[i]:
                expo_count+=1
                edge_exploration_during_training.append(end[i] - init[i])
        self.edge_exploration_times=[]
        self.t=1
        for link in optimal_links:
            optimal_edges_delay_difference_after_training.append(abs(self.Dict_edge_theta[link] - G[link[0]][link[1]]["delay_mean"]))
        self.plotter.plot_edge_delay_difference_for_some_edges(optimal_edges_delay_difference_after_inti,optimal_edges_delay_difference_after_training)
        average_computed_edge_num = sum(computed_edge_num) / len(computed_edge_num)
        #compute the last 1000 correct select
        #optimal_path_selected_rate=sum(correct_shortest_path_selected_rate[-1000:])/1000
        #return rewards_mse_list,selected_shortest_path, expo_count, total_mse_array, edge_exploration_during_training, average_computed_edge_num,optimal_path_selected_rate, avg_diff_of_delay_from_optimal
        return rewards_mse_list, selected_shortest_path, expo_count, total_mse_array, edge_exploration_during_training, average_computed_edge_num, average_optimal_path_selected_rate, avg_diff_of_delay_from_optimal,average_optimal_path_selected_rate_among_monitor_pairs
    def compute_rewards_mse(self,total_rewards_dict, optimal_delay_dict):
        key_list = list(total_rewards_dict.keys())
        rewards_mse_list = []
        sum_rewards_Dict = {}
        time_average_rewards_Dict = {}
        for key in key_list:
            sum_rewards_Dict[key] = 0
            time_average_rewards_Dict[key] = 0
            sum_square = 0
        for i in range(len(total_rewards_dict[key_list[0]])):
            for key in key_list:
                '''solution 1'''
                #sum_rewards_Dict[key] += total_rewards_dict[key][i]
                #time_average_rewards_Dict[key] = sum_rewards_Dict[key] / (i + 1)
                #sum_square += (time_average_rewards_Dict[key] - optimal_delay_dict[key]) ** 2
                '''sulution 2'''
                sum_square += (total_rewards_dict[key][i] - optimal_delay_dict[key]) ** 2
                '''solution 3'''
                #sum_square += abs(total_rewards_dict[key][i] - optimal_delay_dict[key])
            rewards_mse_list.append(sum_square / len(key_list)/(i+1))
        return rewards_mse_list

    def compute_regret(self, total_rewards_dict, optimal_delay_dict):
        key_list = list(total_rewards_dict.keys())
        sum_rewards_Dict = {}
        average_regret_list = []
        for key in key_list:
            sum_rewards_Dict[key] = 0
            # time_average_rewards_Dict[key] = 0
        for i in range(len(total_rewards_dict[key_list[0]])):
            regret_list = []
            for key in key_list:
                sum_rewards_Dict[key] += total_rewards_dict[key][i]
                regret = sum_rewards_Dict[key] - (i + 1) * optimal_delay_dict[key]
                # self.logger.info("regret: %f" %(regret))
                regret_list.append(regret)
            # self.logger.info("regret_list: %s: " %regret_list)
            average_regret_list.append(sum(regret_list) / math.log(i + 2))
            # time_averaged_regret_list.append(sum(regret_list)/len(key_list)/(i+1))
            #self.logger.info("average_regret_list: %s" % average_regret_list)
        return average_regret_list


    def plot_edge_delay_difference_at_different_time_point(self, G):
        if self.t == 1000:
            delay_difference1 = []
            for edge in G.edges:
                delay_difference1.append(abs(self.Dict_edge_theta[edge] - G[edge[0]][edge[1]]['delay_mean']))
            self.edge_delay_difference_list.append(delay_difference1)
            self.plotter.plot_edge_delay_difference(G, self.Dict_edge_theta)
            plt.savefig(self.directory + 'delay difference from mean at t=1000', format="PNG", dpi=300,
                        bbox_inches='tight')
            plt.close()
        if self.t == 2000:
            delay_difference2 = []
            for edge in G.edges:
                delay_difference2.append(abs(self.Dict_edge_theta[edge] - G[edge[0]][edge[1]]['delay_mean']))
            self.edge_delay_difference_list.append(delay_difference2)
            self.plotter.plot_edge_delay_difference(G, self.Dict_edge_theta)
            plt.savefig(self.directory + 'delay difference from mean at t=2000', format="PNG", dpi=300,
                        bbox_inches='tight')
            plt.close()
        if self.t == 3000:
            delay_difference3 = []
            for edge in G.edges:
                delay_difference3.append(abs(self.Dict_edge_theta[edge] - G[edge[0]][edge[1]]['delay_mean']))
            self.edge_delay_difference_list.append(delay_difference3)
            self.plotter.plot_edge_delay_difference(G, self.Dict_edge_theta)
            plt.savefig(self.directory + 'delay difference from mean at t=3000', format="PNG", dpi=300,
                        bbox_inches='tight')
            plt.close()





















