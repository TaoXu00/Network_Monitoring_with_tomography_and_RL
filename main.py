import networkx as nx
import numpy as np
from random import sample
from itertools import combinations
import datetime
import multi_armed_bandit
import network_topology_construction as topology
import network_tomography as tomography
import logging
import os
import plotter as plotter
import sys
class main:
    def __init__(self, time):
        basename="log"
        suffix=datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        self.directory='temp/'+"_".join([basename,suffix])+'/'
        print("Results saved in %s +myapp.log" %(self.directory))
        os.makedirs(self.directory)
        self.trimedGraph_Dir=self.directory+"trimed_Graph/"
        os.makedirs(self.trimedGraph_Dir)
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
        self.MAB=multi_armed_bandit.multi_armed_bandit(self.topo, logger_mab,self.directory) #pass the topology
        self.time=time
        self.plotter = plotter.plotter(self.directory)


    def creat_topology(self,type,n,p):
        #create a network topology
        G= self.topo.graph_Generator(type, n, p)
        return G


    def run_MAB(self, G, monitors, path_sapce):
        self.MAB.Initialize(G, monitors, path_sapce)
        #self.MAB.Initialize_simple(G, monitors)
        monitor_pair_list = list(combinations(monitors, 2))
        optimal_path_dict={}
        optimal_delay_dict={}
        for monitor_pair in monitor_pair_list:
            optimal_path = nx.shortest_path(G, monitor_pair[0], monitor_pair[1], weight='delay_mean', method='dijkstra')
            optimal_delay= nx.path_weight(G, optimal_path, 'delay_mean')
            optimal_path_dict[monitor_pair]=optimal_path
            optimal_delay_dict[monitor_pair]=optimal_delay
        rewards_mse_list, selected_shortest_path, optimal_path_selected_rate, avg_diff_of_delay_from_optimal,  average_n_probing_links,  path_oscilation_list,traffic_overhead_every_200_iterations= self.MAB.train_llc(G, self.time,monitor_pair_list)

        path_dict = {}
        for path in selected_shortest_path:
            p = '-'.join(path)
            if p in path_dict:
                path_dict[p] += 1
            else:
                path_dict[p] = 1
        #self.logger_main.info("paths are explored during the training: %s" %(selected_shortest_path))

        return rewards_mse_list, optimal_delay, optimal_path_selected_rate, avg_diff_of_delay_from_optimal,  average_n_probing_links, path_oscilation_list,traffic_overhead_every_200_iterations

    def MAB_with_increasing_monitors(self, G, type, node_num, p, path_space):
        '''
        In the system configuration, we random created a topology with 100 nodes.
        :param G: the topology graph
        :return: a figure named "network tomography.png" will be saved to show the rate of the identified edges will be
                 increased as the growth of the deployed monitor. it eventually will reach to 1 when the number of the deployed
                 monitor is equal to the number of the edges.
        '''

        monitors_list = []
        end_nodes=[]
        total_rewards_mse_list=[]
        optimal_path_selected_percentage_list = []
        avg_diff_of_delay_from_optimal_list = []
        degree_list = list(G.degree(list(G.nodes)))

        #it does not make sense to differenciate the end nodes from the internal nodes.
        for edge_degree in degree_list:
            if edge_degree[1] == 2 or edge_degree[1]==1:
                end_nodes.append(edge_degree[0])
        #self.logger_main.debug("degree_list: %s" %(degree_list))
        #self.logger_main.debug("end nodes list:%s" %(end_nodes))
        #for n in range(2, len(monitor_candidate_list) + 1, 1):
        #for n in range(2, 3, 1):
        monitors=[]
        monitors_deployment_percentage=[]
        average_n_probing_links_with_increasing_monitors=[]
        path_oscilation_list_with_increasing_monitors = []
        traffic_overhead_every_200_iterations_with_increasing_monitors = []
        for m_p in [10, 20, 30, 40, 50]:
        #for m_p in [30]:
            monitors_deployment_percentage.append(m_p)
            n=int((m_p/100)*len(G.nodes))
            if n==2:
                n=3
            #self.logger_main.debug(f"m_p {m_p}")
            if n <= len(end_nodes):
                rest_end_nodes = [elem for elem in end_nodes if elem not in monitors]
                #self.logger_main.debug(f"rest node {rest_end_nodes}")
                select = sample(rest_end_nodes, k=n - len(monitors))
                #self.logger_main.debug(f"select {select}")
                monitors = monitors + select
            else:
                monitors = self.topo.deploy_monitor(G, n, end_nodes)
            monitors = self.topo.deploy_monitor(G, n, monitors)
            #monitors=['45', '32', '28', '46', '29', '24', '36', '44', '42', '37']
            self.logger_main.info("deloy %d monitors: %s" %(n,monitors))
            trimedG=G
            #trimedG=mynetwork.topo.trimNetwrok(G, monitors)
            nx.write_gml(G, "%sGraph_%s_%s.gml" %(self.trimedGraph_Dir,type,str(m_p)))
            #self.MAB.Initialize(trimedG, monitors)

            rewards_mse_list, optimal_delay, optimal_path_selected_rate, avg_diff_of_delay_from_optimal, average_n_probing_links, path_oscilation_list,traffic_overhead_every_200_iterations=self.run_MAB(trimedG, monitors, path_space)
            monitors_list.append(monitors)
            total_rewards_mse_list.append(rewards_mse_list)
            optimal_path_selected_percentage_list.append(optimal_path_selected_rate)
            avg_diff_of_delay_from_optimal_list.append(avg_diff_of_delay_from_optimal)
            average_n_probing_links_with_increasing_monitors.append(average_n_probing_links)
            path_oscilation_list_with_increasing_monitors.append(path_oscilation_list)
            traffic_overhead_every_200_iterations_with_increasing_monitors.append(traffic_overhead_every_200_iterations)
            self.logger_main.info("percentage of the optimal path selected: %f" % (optimal_path_selected_rate))
            self.logger_main.info(" abs diff from the real optimal path: %f" % (avg_diff_of_delay_from_optimal))
            self.logger_main.info(" %s pert is done, current shape of rate_of_path_oscilatoion_with_increasing_monitors %s" % (m_p, np.array(path_oscilation_list_with_increasing_monitors).shape))
            self.topo.draw_edge_delay_sample(G,type,node_num,p)

        return optimal_path_selected_percentage_list, avg_diff_of_delay_from_optimal_list, monitors_deployment_percentage, avg_diff_of_delay_from_optimal_list, average_n_probing_links_with_increasing_monitors, path_oscilation_list_with_increasing_monitors, traffic_overhead_every_200_iterations_with_increasing_monitors

'''
argv1: network topology type
argv2: number of nodes
argv3: degree of new added nodes in Barabasi network
argv4: enable MAB (1 enable, 0 disable)
'''
if len(sys.argv)!=6:
    raise ValueError('missing parameters')
topo_type=sys.argv[1]
num_node=int(sys.argv[2])
degree=int(sys.argv[3])
num_run=float(sys.argv[4])
path_space=int(sys.argv[5])
print(topo_type, num_node, degree, num_run)

multi_times_optimal_path_selected_percentage_list=[]
multi_times_avg_diff_of_delay_from_optimal_list=[]
multi_times_avg_path_oscilation_list=[]
multi_times_avg_traffic_overhead_every_200_iterations_with_increasing_monitors=[]
n=num_run
i=0
while(i<n):
    mynetwork=main(3000)
    G =mynetwork.creat_topology(topo_type, num_node, degree)
    #mynetwork.tomography_verification(G,'weight')   #here the assigned delay should be 1, place modify the topo.assign_link_delay() function
    optimal_path_selected_percentage_list, avg_diff_of_delay_from_optimal_list, monitors_deployment_percentage, avg_diff_of_delay_from_optimal_list, average_n_probing_links_with_increasing_monitors, path_oscilation_list,traffic_overhead_every_200_iterations =mynetwork.MAB_with_increasing_monitors(G,topo_type,len(G.nodes),degree,path_space)
    if i==0:
        multi_times_optimal_path_selected_percentage_array=np.array([optimal_path_selected_percentage_list])
        multi_times_avg_diff_of_delay_from_optimal_array=np.array([avg_diff_of_delay_from_optimal_list])
        multi_times_avg_n_probing_links_reduced_array = np.array([average_n_probing_links_with_increasing_monitors])
        multi_times_avg_path_oscilation_list=np.array(path_oscilation_list)
        multi_times_avg_traffic_overhead_every_200_iterations_with_increasing_monitors=np.array(traffic_overhead_every_200_iterations)
    else:
        mynetwork.logger_main.info(multi_times_optimal_path_selected_percentage_array)
        multi_times_optimal_path_selected_percentage_array=np.append(multi_times_optimal_path_selected_percentage_array,np.array([optimal_path_selected_percentage_list]), axis=0)
        multi_times_avg_diff_of_delay_from_optimal_array=np.append(multi_times_avg_diff_of_delay_from_optimal_array,np.array([avg_diff_of_delay_from_optimal_list]), axis=0)
        multi_times_avg_n_probing_links_reduced_array = np.append(multi_times_avg_n_probing_links_reduced_array,np.array([average_n_probing_links_with_increasing_monitors]))
        multi_times_avg_path_oscilation_list=np.add(multi_times_avg_path_oscilation_list,path_oscilation_list)
        multi_times_avg_traffic_overhead_every_200_iterations_with_increasing_monitors =np.add(multi_times_avg_traffic_overhead_every_200_iterations_with_increasing_monitors,np.array(traffic_overhead_every_200_iterations))

    i +=1

mynetwork.logger_main.info("Statistics:")
mynetwork.logger_main.info("Original: percentage of the optimal path selected:")
mynetwork.logger_main.info(multi_times_optimal_path_selected_percentage_array)
mynetwork.logger_main.info("Original: diff from the real optimal path:")
mynetwork.logger_main.info(multi_times_avg_diff_of_delay_from_optimal_array)
mynetwork.logger_main.info("original: num of links probed in the selected paths")
mynetwork.logger_main.info(multi_times_avg_n_probing_links_reduced_array)



multi_avg_percentage_of_select_optimal_path=np.average(multi_times_optimal_path_selected_percentage_array,axis=0)
multi_avg_percentage_of_abs_diff_from_optimal=np.average(multi_times_avg_diff_of_delay_from_optimal_array,axis=0)
multi_avg_n_probing_links_with_increasing_monitors=np.average(multi_times_avg_n_probing_links_reduced_array,axis=0)
multi_times_avg_path_oscilation_list=multi_times_avg_path_oscilation_list/n
multi_times_avg_traffic_overhead_every_200_iterations_with_increasing_monitors=multi_times_avg_traffic_overhead_every_200_iterations_with_increasing_monitors/n
np.savetxt("UBC1_ocsillition_every_200_times_BR50_10%-50%_baseline.txt",multi_times_avg_path_oscilation_list)
np.savetxt("UBC1_avg_probing_links_with_increasing_monitors.txt", multi_times_avg_n_probing_links_reduced_array)
np.savetxt("UBC1_traffic_overhead_with_increasing_monitors.txt",  multi_times_avg_traffic_overhead_every_200_iterations_with_increasing_monitors)
mynetwork.logger_main.info("Average: percentage of the optimal path selected:")
mynetwork.logger_main.info(multi_avg_percentage_of_select_optimal_path)
mynetwork.logger_main.info("Average: diff from the real optimal path:")
mynetwork.logger_main.info(multi_avg_percentage_of_abs_diff_from_optimal)
mynetwork.logger_main.info("Average: num of links probed in the selected paths")
mynetwork.logger_main.info(multi_avg_n_probing_links_with_increasing_monitors)
mynetwork.logger_main.info("Average: path oscillitation")
mynetwork.logger_main.info(multi_times_avg_path_oscilation_list)
np.savetxt("UCB1_oscillation_BTN.txt",multi_times_avg_path_oscilation_list)
mynetwork.logger_main.info("Average: traffic overhead every 200 iterations")
mynetwork.logger_main.info(multi_times_avg_traffic_overhead_every_200_iterations_with_increasing_monitors)
mynetwork.plotter.plot_optimal_path_selected_percentage_list_with_increasing_monitors(monitors_deployment_percentage, multi_avg_percentage_of_select_optimal_path)
mynetwork.plotter.plot_abs_diff_path_delay_from_the_optimal(monitors_deployment_percentage,multi_avg_percentage_of_abs_diff_from_optimal )
mynetwork.plotter.plot_avg_path_oscilation_every_200_times(monitors_deployment_percentage, multi_times_avg_path_oscilation_list)
mynetwork.plotter.plot_avg_traffic_overhead_every_200_iterations(monitors_deployment_percentage,multi_times_avg_traffic_overhead_every_200_iterations_with_increasing_monitors)