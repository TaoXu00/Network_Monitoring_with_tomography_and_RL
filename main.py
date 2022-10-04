import networkx as nx
import numpy as np
from random import sample
import sys
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

    def run_MAB(self, G, monitors, llc_factor):
        self.MAB.Initialize(G, monitors)
        monitor_pair_list = list(combinations(monitors, 2))
        optimal_path_dict={}
        optimal_delay_dict={}
        for monitor_pair in monitor_pair_list:
            optimal_path = nx.shortest_path(G, monitor_pair[0], monitor_pair[1], weight='delay_mean', method='dijkstra')
            optimal_delay= nx.path_weight(G, optimal_path, 'delay_mean')
            optimal_path_dict[monitor_pair]=optimal_path
            optimal_delay_dict[monitor_pair]=optimal_delay
        rewards_mse_list, selected_shortest_path, expo_count,total_mse_array,edge_exploration_during_training, average_computed_edge_num, optimal_path_selected_rate, avg_diff_of_delay_from_optimal = self.MAB.train_llc(G, self.time,monitor_pair_list, llc_factor)

        path_dict = {}
        for path in selected_shortest_path:
            p = '-'.join(path)
            if p in path_dict:
                path_dict[p] += 1
            else:
                path_dict[p] = 1
        #self.logger_main.info("paths are explored during the training: %s" %(path_dict))
        return expo_count, total_mse_array, rewards_mse_list, optimal_delay, edge_exploration_during_training, average_computed_edge_num, optimal_path_selected_rate, avg_diff_of_delay_from_optimal

    def MAB_with_increasing_monitors(self, G, type, node_num, p, llc_factor):
        '''
        In the system configuration, we random created a topology with 100 nodes.
        :param G: the topology graph
        :return: a figure named "network tomography.png" will be saved to show the rate of the identified edges will be
                 increased as the growth of the deployed monitor. it eventually will reach to 1 when the number of the deployed
                 monitor is equal to the number of the edges.
        '''

        monitors_list = []
        explored_edges_rate = []
        end_nodes=[]
        total_edge_mse_list_with_increasing_monitors = []
        total_edge_exploration_during_training_list = []
        average_computed_edge_rate_during_training = []
        total_rewards_mse_list=[]
        optimal_path_selected_percentage_list=[]
        optimal_path_select_rate_amoong_monitors_list=[]
        avg_diff_of_delay_from_optimal_list=[]
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
        for m_p in [30]:
        #for m_p in [20, 30]:
            monitors_deployment_percentage.append(m_p)
            n = int((m_p / 100) * len(G.nodes))
            if n <= len(end_nodes):
                rest_end_nodes = [elem for elem in end_nodes if elem not in monitors]
                # self.logger_main.debug(f"rest node {rest_end_nodes}")
                select = sample(rest_end_nodes, k=n - len(monitors))
                # self.logger_main.debug(f"select {select}")
                monitors = monitors + select
            else:
                monitors = self.topo.deploy_monitor(G, n, end_nodes)
            monitors = self.topo.deploy_monitor(G, n, monitors)
            # monitors=['45', '32', '28', '46', '29', '24', '36', '44', '42', '37']
            self.logger_main.info("deloy %d pert monitors: %s" % (m_p, monitors))
            trimedG=mynetwork.topo.trimNetwrok(G, monitors)
            #trimedG = G
            nx.write_gml(trimedG, "%sGraph_%s_%s.gml" % (self.trimedGraph_Dir, type, str(m_p)))

            expo_count, total_mse, rewards_mse_list, optimal_delay, edge_exploration_during_training, average_computed_edge_num, optimal_path_selected_rate, avg_diff_of_delay_from_optimal = self.run_MAB(
                trimedG, monitors, llc_factor)
            monitors_list.append(monitors)
            explored_edges_rate.append(expo_count / len(trimedG.edges))
            total_edge_mse_list_with_increasing_monitors.append(total_mse)
            total_rewards_mse_list.append(rewards_mse_list)
            total_edge_exploration_during_training_list.append(edge_exploration_during_training)
            average_computed_edge_rate_during_training.append(average_computed_edge_num / len(trimedG.edges))
            optimal_path_selected_percentage_list.append(optimal_path_selected_rate)
            avg_diff_of_delay_from_optimal_list.append(avg_diff_of_delay_from_optimal)

            self.logger_main.info("edges explored: %f" % (expo_count / len(trimedG.edges)))
            self.logger_main.info("edges computed: %f" % (average_computed_edge_num / len(trimedG.edges)))
            # np.savetxt("mse_with_NT_in_training_node%s.txt" %(len(G.nodes)), np_array_total_mse, delimiter=",")
            self.logger_main.info("percentage of the optimal path selected: %f" % (optimal_path_selected_rate))
            self.logger_main.info(" abs diff from the real optimal path: %f" %(avg_diff_of_delay_from_optimal))
            self.topo.draw_edge_delay_sample(G, type, node_num, p)

        arr = np.array(average_computed_edge_rate_during_training)
        np.savetxt('%sidentificable edges rate with increasing monitors' % (self.directory), arr)
        # self.plotter.plot_rewards_mse_along_with_different_monitors(monitors_deployment_percentage,total_rewards_mse_list)
        # self.plotter.plot_bar_edge_exploration_training_with_increasing_monitor(monitors_deployment_percentage, explored_edges_rate)
        self.plotter.plot_edge_computed_rate_during_training(monitors_deployment_percentage, average_computed_edge_rate_during_training)
        return optimal_path_selected_percentage_list, avg_diff_of_delay_from_optimal_list, total_edge_mse_list_with_increasing_monitors,monitors_deployment_percentage

    def plot_edge_computed_rate_bar_with_different_topology_size(self):
        self.plotter.plot_edge_computed_rate_with_different_topology_size()

    def plot_final_result(self, mynetwork):
        #plot the percentage of optimal path selected in BR
        '''
        monitors_deployment_percentage = [10, 20, 30, 40, 50]
        myapproach_optimal_path_selected_rate = [0.7303425, 0.784525, 0.820643333, 0.838918553, 0.902852585]
        baseline_optimal_path_selected_rate = [0.5381625, 0.588136115, 0.556531905, 0.57743605, 0.621137165]
        baseline_abs_of_optimal_path_selected_from_real = [4.607487055, 4.237622065, 4.815920555, 5.09833035, 5.258270955]
        myapproach_abs_of_optimal_path_selected_from_real = [3.62672339, 2.700724235, 2.368452608, 2.251432833, 1.676121458]
        mynetwork.plotter.plot_percentage_of_optimal_path_selected_rate_BR_50nodes(monitors_deployment_percentage, myapproach_optimal_path_selected_rate, baseline_optimal_path_selected_rate)
        mynetwork.plotter.plot_abs_delay_of_optimal_path_selected_from_mean_BR_50nodes(monitors_deployment_percentage, myapproach_abs_of_optimal_path_selected_from_real, baseline_abs_of_optimal_path_selected_from_real)
        '''
        topology_size=[20, 40, 60, 80]
        myapproach_optimal_path_selected_rate=[0.86105667,0.7691784,0.723593333,0.58820978]
        baseline_optimal_path_selected_rate=[0.64950667,0.56473333,	0.56776732,	0.57303025]
        baseline_abs_of_optimal_path_selected_from_real=[3.54004911, 4.82165252, 4.89638844, 5.31785722]
        myapproach_abs_of_optimal_path_selected_from_real=[1.66976496, 2.97261583,	2.514051111, 3.88283325]
        mynetwork.plotter.plot_percentage_of_optimal_path_selected_rate_for_various_monitor_size(topology_size, myapproach_optimal_path_selected_rate, baseline_optimal_path_selected_rate)
        mynetwork.plotter.plot_abs_delay_of_optimal_path_selected_for_various_monitor_size(topology_size, myapproach_abs_of_optimal_path_selected_from_real, baseline_abs_of_optimal_path_selected_from_real)
       #plot the experiments for real infrastructure
        monitors_deployment_percentage = [10, 20, 30, 40, 50]
        myapproach_optimal_path_selected_rate = [0.8730325,	0.865852223, 0.84571643, 0.835979885,0.843303463]
        baseline_optimal_path_selected_rate = [0.67076,	0.68020833,	0.67219667,	0.66329536,	0.67381338]
        baseline_abs_of_optimal_path_selected_from_real = [3.92730099,	3.55005621,	3.76985005,	4.01648821,	4.05722965]
        myapproach_abs_of_optimal_path_selected_from_real = [1.548090993,1.607211863,1.779516288,1.887000793,1.844019875]
        mynetwork.plotter.plot_percentage_of_optimal_path_selected_rate_BTN(monitors_deployment_percentage, myapproach_optimal_path_selected_rate, baseline_optimal_path_selected_rate)
        mynetwork.plotter.plot_abs_delay_of_optimal_path_selected_from_mean_BTN(monitors_deployment_percentage, myapproach_abs_of_optimal_path_selected_from_real, baseline_abs_of_optimal_path_selected_from_real)

'''
argv1: network topology type
argv2: number of nodes
argv3: degree of new added nodes in Barabasi network
argv4: enable MAB (1 enable, 0 disable)
'''

if len(sys.argv)!=6:
    print(len(sys.argv))
    raise ValueError('missing parameters')
topo_type=sys.argv[1]
num_node=int(sys.argv[2])
degree=int(sys.argv[3])
llc_factor=float(sys.argv[4])
num_run=int(sys.argv[5])
print(topo_type, num_node, degree, llc_factor, num_run)

multi_times_optimal_path_selected_percentage_list=[]
multi_times_avg_diff_of_delay_from_optimal_list=[]
n=num_run
i=0
'''
mynetwork=main(3000)
mynetwork.plot_final_result(mynetwork)
'''

while(i<n):
    mynetwork=main(3000)
    G =mynetwork.creat_topology(topo_type, num_node, degree)
    #mynetwork.tomography_verification(G,'weight')   #here the assigned delay should be 1, place modify the topo.assign_link_delay() function
    optimal_path_selected_percentage_list, avg_diff_of_delay_from_optimal_list,total_edge_mse_list_with_increasing_monitors,monitors_deployment_percentage = mynetwork.MAB_with_increasing_monitors(G,topo_type,len(G.nodes),degree, llc_factor)
    #print("n=%d" %(i))
    #print(optimal_path_selected_percentage_list,avg_diff_of_delay_from_optimal_list)
    if i==0:
        multi_times_mse_total_link_delay_array=np.array(total_edge_mse_list_with_increasing_monitors, dtype=float)
        multi_times_optimal_path_selected_percentage_array=np.array([optimal_path_selected_percentage_list])
        multi_times_avg_diff_of_delay_from_optimal_array=np.array([avg_diff_of_delay_from_optimal_list])
    else:
        current_mse_arrary=np.array(total_edge_mse_list_with_increasing_monitors)
        multi_times_mse_total_link_delay_array=np.add(multi_times_mse_total_link_delay_array,current_mse_arrary)
        multi_times_optimal_path_selected_percentage_array=np.append(multi_times_optimal_path_selected_percentage_array,np.array([optimal_path_selected_percentage_list]),axis=0)
        multi_times_avg_diff_of_delay_from_optimal_array=np.append(multi_times_avg_diff_of_delay_from_optimal_array,np.array([avg_diff_of_delay_from_optimal_list]), axis=0)

    i += 1


multi_times_avg_mse_total_link_delay_array=multi_times_mse_total_link_delay_array/n
np.savetxt(mynetwork.directory + 'links_delay_during_training_with_different_monitor_size.txt', multi_times_avg_mse_total_link_delay_array)
mynetwork.logger_main.info("Statistics:")
mynetwork.logger_main.info("Before average: percentage of the optimal path selected:")
mynetwork.logger_main.info(multi_times_optimal_path_selected_percentage_array)
mynetwork.logger_main.info("Before average: diff from the real optimal path: ")
mynetwork.logger_main.info(multi_times_avg_diff_of_delay_from_optimal_array)



multi_avg_percentage_of_select_optimal_path=np.average(multi_times_optimal_path_selected_percentage_array,axis=0)
multi_avg_percentage_of_abs_diff_from_optimal=np.average(multi_times_avg_diff_of_delay_from_optimal_array,axis=0)
mynetwork.logger_main.info("after average: percentage of the optimal path selected:")
mynetwork.logger_main.info (multi_avg_percentage_of_select_optimal_path)
mynetwork.logger_main.info("after average: diff from the real optimal path:")
mynetwork.logger_main.info(multi_avg_percentage_of_abs_diff_from_optimal)


mynetwork.plotter.plot_total_edge_delay_mse_with_increasing_monitor_training(monitors_deployment_percentage,multi_times_avg_mse_total_link_delay_array)
# self.plotter.plot_edge_exporation_times_with_differrent_monitor_size(G,total_edge_exploration_during_training_list)
mynetwork.plotter.plot_optimal_path_selected_percentage_list_with_increasing_monitors(monitors_deployment_percentage, multi_avg_percentage_of_select_optimal_path)
mynetwork.plotter.plot_abs_diff_path_delay_from_the_optimal(monitors_deployment_percentage,multi_avg_percentage_of_abs_diff_from_optimal )
'''test
array1=np.array([[0.1, 0.2, 0.3],
                [0.2, 0.3, 0.4],
                [0.4, 0.5, 0.6]])
array2=np.array([[0.2, 0.3, 0.4],
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6]])
sum=np.add(array1,array2)
three_avg=sum/3
print(three_avg)
'''