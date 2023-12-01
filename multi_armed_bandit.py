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
        plt.close()


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
                #self.logger.error(str(e)+"occurred, try the inversed direction.")
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
        self.logger.debug(f"x={x}")
        self.logger.debug(f"explored_edge_set:{explored_edge_set}")
        for i in range(len(x)):
            self.logger.debug(f"x{i}={x[i]}")
            if x[i] != 0:
                edge = edges[i]
                #self.logger.debug(
                #    "edge %s is computed with the value %f, its realtime delay is %f " %(edge, x[i],G[edge[0]][edge[1]]['delay']))
                self.Dict_edge_theta[edge] = (self.Dict_edge_theta[edge] * self.Dict_edge_m[edge] + x[i]) / (
                        self.Dict_edge_m[edge] + 1)
                self.Dict_edge_m[edge] = self.Dict_edge_m[edge] + 1
                if edge not in explored_edge_set:
                    edge = (edge[1], edge[0])
                self.logger.debug(f"going to remove edge {edge}")
                explored_edge_set.remove(edge)
                self.logger.debug(f"removed edge {i}")
                self.logger.debug(f"explored_edge_set:{explored_edge_set}")

        self.logger.info(f"edges to be updated: {explored_edge_set}")
        for edge in explored_edge_set:
            if edge not in edges:
                edge_g = (edge[1], edge[0])
            else:
                edge_g=edge
            #as a base line approach, the unindentificable links will be assigned with average delay(total_path_delay/path_length)
            self.logger.debug(f"{edge_g} is unindentified, update its value with the estimation approach with value:{edge_average_delay_Dict[edge_g]}")
            self.Dict_edge_theta[edge_g] = (self.Dict_edge_theta[edge_g] * self.Dict_edge_m[edge_g] + edge_average_delay_Dict[edge_g]) / (
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
        #print(llc)
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
        optimal_links=[]   #store the links that only appears in the optimal path
        for key in optimal_path_dict:
            path = optimal_path_dict[key]
            path_pair = self.construct_pathPair_from_path(path)
            for pair in path_pair:
                if pair not in optimal_links:
                    optimal_links.append(pair)
        selected_shortest_path=[]
        total_mse_array = []
        total_mse_optimal_edges_array=[]
        total_rewards_dict = {}   #in the current implementation, it is for only one pair of monitors
        computed_edge_num=[]
        correct_shortest_path_selected_rate = []
        optimal_edges_delay_difference_after_inti=[]
        optimal_edges_delay_difference_after_training=[]
        for link in optimal_links:
            optimal_edges_delay_difference_after_inti.append(abs(self.Dict_edge_theta[link] - G[link[0]][link[1]]["delay_mean"]))
        self.logger.debug("%d optimal links: %s" %(len(optimal_links), optimal_links))
        self.logger.debug("start trainning...")
        for monitor_pair in monitor_pair_list:
            total_rewards_dict[monitor_pair]=[]
        diff_of_delay_from_optimal_real_time=[]
        Dict_time_of_optimal_path_selected = {}
        rate_optimal_path_selected = []
        for monitor_pair in monitor_pair_list:
            Dict_time_of_optimal_path_selected[monitor_pair] = []
        sum_n_links_origin =0
        sum_n_links_reduced=0
        sum_n_links_reduced_random=0
        rate_of_optimal_actions_list=[]
        path_oscilation_list=[]
        counter=0
        dict_n_paths = {}
        traffic_overhead_every_200_iterations = []
        dict_theta_along_time={}
        dict_inferred_from_real={}
        for edge in G.edges:
            dict_inferred_from_real[edge] = []
        e2e_avg_overtime=[]

        dict_edge_in_paths_overtime={}
        for i in range(time):
            ##compute the mse of all the links in the graph during training
            self.logger.info("t= %s" %(self.t))
            total_mse = 0
            total_mse_opt_edges = 0

            diff_from_true_mean=[]
            Dict_edge_in_paths = {}  # the key is the edges and the values are the number paths including this link, and the new_theta = new_theta= theta + #path* E%*theta
            for edge in G.edges:
                if i==0:
                    dict_theta_along_time[edge]=[self.Dict_edge_theta[edge]]
                else:
                    dict_theta_along_time[edge].append(self.Dict_edge_theta[edge])
                total_mse += (self.Dict_edge_theta[edge] - G[edge[0]][edge[1]]['delay_mean']) ** 2
                diff_from_true_mean.append(self.Dict_edge_theta[edge] - G[edge[0]][edge[1]]['delay_mean'])
                if edge in optimal_links:
                    total_mse_opt_edges+= (self.Dict_edge_theta[edge] - G[edge[0]][edge[1]]['delay_mean']) ** 2
                #every iteration,the path number should be reset.
                Dict_edge_in_paths[edge] = 0
            total_mse_array.append(total_mse / len(G.edges))
            total_mse_optimal_edges_array.append(total_mse_opt_edges/len(optimal_links))
            explored_path_list = []
            total_diff=0
            optimal_actions=0
            #inilization of the paths dict
            if counter==0:
                for monitor_pair in monitor_pair_list:
                    dict_n_paths[monitor_pair] =set()
                    traffic_overhead_in_200_iterations = []
            if counter==200:  #200 iterations, add the path number to a list
                #iterate the dictionary
                sum_paths=0
                for key in dict_n_paths:
                    sum_paths+=len(dict_n_paths[key])
                avg=sum_paths/len(monitor_pair_list)
                path_oscilation_list.append(avg)
                avg_200_iteration = np.average(traffic_overhead_in_200_iterations)
                traffic_overhead_every_200_iterations.append(avg_200_iteration)
                counter=0
                traffic_overhead_in_200_iterations=[]
                for monitor_pair in monitor_pair_list:
                    dict_n_paths[monitor_pair] =set()
                #self.logger.debug("current path oscilation_list:%s" %(path_oscilation_list))
                #self.logger.debug("reset the dict: %s" %(dict_n_paths))
            #self.logger.debug(dict_n_paths)
            for monitor_pair in monitor_pair_list:
                m1=monitor_pair[0]
                m2=monitor_pair[1]
                shortest_path=self.LLC_policy(G, m1, m2, llc_factor)
                if shortest_path==optimal_path_dict[monitor_pair]:
                    optimal_actions+=1
                self.logger.debug(f"shortest path for {monitor_pair} is {shortest_path}")
                dict_n_paths[monitor_pair].add(tuple(shortest_path))
                #shortest_path = self.LLC_policy_without_MAB(G, m1, m2)
                #collect the paths it used in 200 iterations

                if shortest_path == optimal_path_dict[monitor_pair] and self.t>=time-1000:
                    Dict_time_of_optimal_path_selected[monitor_pair].append(1)
                else:
                    Dict_time_of_optimal_path_selected[monitor_pair].append(0)
                #else:  #check how far it is different from the real optimal path
                    #total_diff+= abs(nx.path_weight(G,shortest_path,"delay_mean")- optimal_delay_dict[monitor_pair])
                total_diff += abs(nx.path_weight(G,optimal_path_dict[monitor_pair], "delay")-nx.path_weight(G, shortest_path, "delay"))
                explored_path_list.append(shortest_path)
                rewards = nx.path_weight(G, shortest_path, 'delay')
                total_rewards_dict[monitor_pair].append(rewards)

            rate_of_optimal_actions_list.append(optimal_actions/len(monitor_pair_list))
            #self.logger.debug("rate_of_optimal_actions_list: %s" %(rate_of_optimal_actions_list))
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
            e2e_avg_overtime.append(np.sum(b)/len(b[0]))

            # for i in range(len(average_delay_list)):
            #     for edge in path_list[i]:
            #         if edge not in edge_average_delay_list_dict:
            #             edge_average_delay_list_dict[edge] = []
            #         edge_average_delay_list_dict[edge].append(average_delay_list[i])
            # for edge in edge_average_delay_list_dict:
            #     edge_average_delay_dict[edge]=sum(edge_average_delay_list_dict[edge])/len(edge_average_delay_list_dict[edge])
            #self.logger.debug("Dict=%s" %(edge_average_delay_dict))
            # get the explored edge set
            explored_edge_set = []
            for path in path_list:
                for edge in path:
                    if (edge[0], edge[1]) not in explored_edge_set and (edge[1], edge[0]) not in explored_edge_set:
                        explored_edge_set.append(edge)
                    #count how many times the path
                    if edge in Dict_edge_in_paths.keys():
                        key=edge
                    else:
                        key=(edge[1], edge[0])
                    Dict_edge_in_paths[key]=Dict_edge_in_paths[key]+1
            #call NT as a submoudle
            x, count,n_links_origin,n_links_reduced, n_links_any_probe_path = self.nt.nt_engine(G, path_list, b)
            edges=list(G.edges)
            unindent_edges=[]
            edge_average_delay_list_dict={}
            for i in range(len(x)):
                if x[i]==0:
                    unindent_edges.append(i)
                    edge_average_delay_list_dict[edges[i]]=[]

            for i in range(len(path_list)):
                print(b.shape)
                sum=b[0][i]
                unident_links=[]
                for edge in path_list[i]:
                    if edge not in edges:
                        edge=(edge[1], edge[0])
                    if edges.index(edge) not in unindent_edges:
                        sum-=x[edges.index(edge)]
                    else:
                        unident_links.append(edge)
                for edge in unident_links:
                    edge_average_delay_list_dict[edge].append(sum/len(unident_links))
            for edge in edge_average_delay_list_dict:
                print(f"dict {edge}: {edge_average_delay_list_dict[edge]}")
                # sum_edge= np.sum(edge_average_delay_list_dict[edge])
                # edge_num= len(edge_average_delay_list_dict[edge])
                edge_average_delay_dict[edge]=np.sum(edge_average_delay_list_dict[edge])/len(edge_average_delay_list_dict[edge])
            sum_n_links_origin+=n_links_origin
            sum_n_links_reduced+=n_links_reduced
            sum_n_links_reduced_random+=n_links_any_probe_path
            traffic_overhead_in_200_iterations.append(n_links_reduced)
            counter+=1
            computed_edge_num.append(count)
            # the MBA variables should be updated according to the results computed by the NT.
            self.update_MBA_variabels_with_NT(G, x, explored_edge_set, edge_average_delay_dict)
            edges = list(G.edges)
            for i in range(len(x)):
                dict_inferred_from_real[edges[i]].append(x[i] - G[edges[i][0]][edges[i][1]]['delay'])

                 # if self.t==15:
                 #     dict_inferred_from_real[edges[i]]=[x[i]-G[edges[i][0]][edges[i][1]]['delay']]
                 # else:
                 #     dict_inferred_from_real[edges[i]].append(x[i]-G[edges[i][0]][edges[i][1]]['delay'])
            '''
            total_rewards.append(rewards)
            regret = sum(total_rewards) - self.t * optimal_delay
            total_regrets.append(regret)
            '''
            self.t = self.t + 1  # the time slot increase 1
            #assign the link delay with a new instance drawn from the updated distribution.
            if self.t < time+len(G.edges):
                #assign a new delay according to the new distribution
                for edge in G.edges:
                    # print(f"Dict_edge_delay_sample: {edge} {self.Dict_edge_delay_sample[edge]}")
                    #dynamic_theta=self.Dict_edge_theta[edge] + Dict_edge_in_paths[edge]*0.009*self.Dict_edge_theta[edge]
                    dynamic_theta = self.Dict_edge_theta[edge] + Dict_edge_in_paths[edge] * 0.008
                    G[edge[0]][edge[1]]['delay'] = np.random.exponential(scale=dynamic_theta)
                    if edge not in dict_edge_in_paths_overtime.keys():
                        link_utility= np.array([Dict_edge_in_paths[edge]])
                        dict_edge_in_paths_overtime[edge]= link_utility
                    else:
                        link_utility=np.append(dict_edge_in_paths_overtime[edge], Dict_edge_in_paths[edge])
                        dict_edge_in_paths_overtime[edge]=link_utility

        #compute the average of the last 200 iterations and see the link utility distribution
        averaged_per_link_utility=[]
        for edge in G.edges:
            a=dict_edge_in_paths_overtime[edge]
            averaged_per_link_utility=np.append(averaged_per_link_utility,np.sum(a[-200:])/200)
        np.savetxt(self.directory+'averaged_per_link_utility_%d_pair_monitors.txt' %(len(monitor_pair_list)),  averaged_per_link_utility)
        #plot the link utility distribution
        plt.figure()
        plt.hist(averaged_per_link_utility,bins=5)
        plt.savefig(self.directory+'link_utility_%d_pair_monitors.png' %(len(monitor_pair_list)), format='png')
        #keep track of the number of selected paths in which each link appears.

        #calculating the averaged e2e delay over time
        e2e_avg_overtime_averages_every_100 = []
        for i in range(0, len(e2e_avg_overtime), 100):
            subset = e2e_avg_overtime[i:i + 100]
            if subset:  # Ensure subset is not empty
                avg = np.sum(subset) / len(subset)
                e2e_avg_overtime_averages_every_100.append(avg)
        np.savetxt(self.directory+'e2e_delay_avg_%d_pair_monitors.txt' %(len(monitor_pair_list)), e2e_avg_overtime_averages_every_100)
        plt.figure()
        plt.plot(e2e_avg_overtime_averages_every_100)
        plt.savefig(self.directory+"e2e_delay_avg_%d_pair_monitors.png" %len(monitor_pair_list), format='png')

        # for key in dict_inferred_from_real:
        #     plt.figure()
        #     plt.plot(np.arange(len(dict_inferred_from_real[key])),dict_inferred_from_real[key])#self.update_MBA_variabels(G,explored_edge_set)
        #     plt.savefig(self.directory+f"{key}_diff_from_real.png", format="png")
        #     #plt.savefig(self.directory+f"{key}_diff_from_real.png", format='png')

        for key in dict_theta_along_time:
            plt.figure()
            plt.plot(np.arange(len(dict_theta_along_time[key])),dict_theta_along_time[key])
            plt.axhline(y = G[key[0]][key[1]]["delay_mean"], color = 'r')
            plt.savefig(self.directory+f"{key}.png", format='png')
        sum_paths = 0
        for key in dict_n_paths:
            sum_paths += len(dict_n_paths[key])
        avg = sum_paths / len(monitor_pair_list)
        path_oscilation_list.append(avg)
        self.logger.debug("current path oscilation_list:%s" %(path_oscilation_list))
        avg_overhead = np.average(traffic_overhead_in_200_iterations)
        traffic_overhead_every_200_iterations.append(avg_overhead)
        self.logger.debug("current traffic overhead list: %s" % (traffic_overhead_every_200_iterations))
        for monitor_pair in monitor_pair_list:
            count_list = Dict_time_of_optimal_path_selected[monitor_pair]
            rate = np.sum(count_list[-1000:])/1000
            #rate = np.sum(count_list[-300:]) / 300
            rate_optimal_path_selected.append(rate)
        average_optimal_path_selected_rate = np.average(np.array(rate_optimal_path_selected))
        correct_shortest_path_selected_rate_array=np.array(correct_shortest_path_selected_rate)
        average_optimal_path_selected_rate_among_monitor_pairs= np.average(np.array(correct_shortest_path_selected_rate_array[-1000:]))
        avg_diff_of_delay_from_optimal = (np.sum(diff_of_delay_from_optimal_real_time) / len(diff_of_delay_from_optimal_real_time))
        rewards_mse_list=self.compute_rewards_mse(total_rewards_dict, optimal_delay_dict)
        average_regret_list = self.compute_regret(total_rewards_dict, optimal_delay_dict)
        self.plotter.plot_total_edge_delay_mse(total_mse_array)
        self.plotter.plot_total_optimal_edge_delay_mse(total_mse_optimal_edges_array)
        self.plotter.plot_time_average_rewards(rewards_mse_list)
        self.plotter.plot_average_regrets(average_regret_list)
        self.plotter.plot_diff_from_optimal_path_of_selected_shortest_paths(diff_of_delay_from_optimal_real_time)
        #plot the delay difference from the mean along time
        #self.plotter.plot_edge_delay_difference_alongtime(0,15,self.edge_delay_difference_list,'0-15')
        #self.plotter.plot_edge_delay_difference_alongtime(15, 30,self.edge_delay_difference_list,'15-30')
        #self.plotter.plot_edge_delay_difference_alongtime(30, 35,self.edge_delay_difference_list, '30-35')

        #self.plotter.plot_edge_exploitation_times_bar('t=3000',self.Dict_edge_m)
        plt.savefig(self.directory + '# of edge exploration after training', format="PNG", dpi=300,
                    bbox_inches='tight')
        plt.close()
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
        # self.logger.debug(f"optimal links: {optimal_links}")
        # self.logger.debug(f"diff from delay mean after training: {optimal_edges_delay_difference_after_training}")
        self.logger.debug(f"opt_edges_delay_after init:{optimal_edges_delay_difference_after_inti}")
        self.logger.debug(f"opt_edges_delay_after training:{optimal_edges_delay_difference_after_training}")
        self.plotter.plot_edge_delay_difference_for_some_edges(optimal_edges_delay_difference_after_inti,optimal_edges_delay_difference_after_training)
        plt.figure()
        plt.plot(np.arange(len(computed_edge_num)),computed_edge_num)
        plt.savefig(self.directory+"edge_count.png", format="PNG")
        average_computed_edge_num = np.sum(computed_edge_num) / len(computed_edge_num)
        average_probing_links_origin=sum_n_links_origin/time
        average_probing_links_reduced=sum_n_links_reduced/time
        average_probing_links_reduced_random=sum_n_links_reduced_random/time
        self.logger.debug("optimal_reduced_links %d" %(average_probing_links_reduced))
        self.logger.debug("random_reduced_links %d" %(average_probing_links_reduced_random))
        #average_computed_edge_num=0
        #compute the last 1000 correct select
        #optimal_path_selected_rate=sum(correct_shortest_path_selected_rate[-1000:])/1000
        #return rewards_mse_list,selected_shortest_path, expo_count, total_mse_array, edge_exploration_during_training, average_computed_edge_num,optimal_path_selected_rate, avg_diff_of_delay_from_optimal
        return rewards_mse_list, selected_shortest_path, expo_count, total_mse_array, total_mse_optimal_edges_array,edge_exploration_during_training, average_computed_edge_num, average_optimal_path_selected_rate, avg_diff_of_delay_from_optimal, average_probing_links_origin, average_probing_links_reduced, rate_of_optimal_actions_list, path_oscilation_list, traffic_overhead_every_200_iterations, e2e_avg_overtime_averages_every_100, averaged_per_link_utility
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

    def update_MBA_variabels_with_NT_test(self, G, x, explored_edge_set, edge_average_delay_Dict):
        edges = list(G.edges)
        self.logger.debug(f"x={x}")
        self.logger.debug(f"explored_edge_set:{explored_edge_set}")
        for i in range(len(x)):
            self.logger.debug(f"x{i}={x[i]}")
            edge = edges[i]
            if edge in explored_edge_set or (edge[1], edge[0]) in explored_edge_set:
                self.logger.debug(
                    "edge %s is computed with the value %f, its realtime delay is %f " %(edge, x[i],G[edge[0]][edge[1]]['delay']))
                diff=x[i]-G[edge[0]][edge[1]]['delay']
                # if diff<1:
                self.Dict_edge_theta[edge] = (self.Dict_edge_theta[edge] * self.Dict_edge_m[edge] + x[i]) / (
                        self.Dict_edge_m[edge] + 1)
                self.Dict_edge_m[edge] = self.Dict_edge_m[edge] + 1
                if edge not in explored_edge_set:
                    edge = (edge[1], edge[0])
                explored_edge_set.remove(edge)
                self.logger.debug(f"removed edge {i}")
                self.logger.debug(f"explored_edge_set:{explored_edge_set}")

        for edge in explored_edge_set:
            if edge not in edges:
                edge_g = (edge[1], edge[0])
            else:
                edge_g=edge
            #as a base line approach, the unindentificable links will be assigned with average delay(total_path_delay/path_length)
            self.logger.debug(f"{edge_g} is unindentified, update its value with the estimation approach with value:{edge_average_delay_Dict[edge]}")
            self.Dict_edge_theta[edge_g] = (self.Dict_edge_theta[edge_g] * self.Dict_edge_m[edge_g] + edge_average_delay_Dict[edge]) / (
                    self.Dict_edge_m[edge_g] + 1)
            self.Dict_edge_m[edge_g] = self.Dict_edge_m[edge_g] + 1
            #self.logger.debug("updating %s with average %s" %(edge, edge_average_delay_Dict[edge]))