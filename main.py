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
import pandas as pd
import scipy.io
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
        self.logger_main.info(f"optimal paths: {optimal_path_dict}")
        self.logger_main.info(f"optimal delay: {optimal_delay_dict}")
        rewards_mse_list, selected_shortest_path, expo_count,total_mse_array,total_mse_optimal_edges_array, edge_exploration_during_training, average_computed_edge_num, optimal_path_selected_rate, avg_diff_of_delay_from_optimal,average_probing_links_origin, average_probing_links_reduced, rate_of_optimal_actions_list, path_oscilation_list, traffic_overhead_every_200_iterations = self.MAB.train_llc(G, self.time,monitor_pair_list, llc_factor)

        path_dict = {}
        for path in selected_shortest_path:
            p = '-'.join(path)
            if p in path_dict:
                path_dict[p] += 1
            else:
                path_dict[p] = 1
        #self.logger_main.info("paths are explored during the training: %s" %(path_dict))
        return expo_count, total_mse_array, total_mse_optimal_edges_array, rewards_mse_list, optimal_delay, edge_exploration_during_training, average_computed_edge_num, optimal_path_selected_rate, avg_diff_of_delay_from_optimal, average_probing_links_origin, average_probing_links_reduced, rate_of_optimal_actions_list, path_oscilation_list, traffic_overhead_every_200_iterations

    def MAB_with_increasing_monitors(self, G, type, node_num, p, llc_factor, monitor_pert_list):
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
        total_optimal_edges_mse_list_with_increasing_monitors =[]
        total_edge_exploration_during_training_list = []
        average_computed_edge_rate_during_training = []
        total_rewards_mse_list=[]
        optimal_path_selected_percentage_list=[]
        optimal_path_select_rate_amoong_monitors_list=[]
        average_probing_links_origin_list=[]
        average_probing_links_reduced_list=[]
        avg_diff_of_delay_from_optimal_list=[]
        degree_list = list(G.degree(list(G.nodes)))
        #it does not make sense to differenciate the end nodes from the internal nodes.
        #trim the node with degree 1
        self.logger_main.info("After trim the degree one nodes: %d nodes %d edges", len(G.nodes), len(G.edges))

        for edge_degree in degree_list:
            #if edge_degree[1] == 2 or edge_degree[1]==1:
                if edge_degree[1] == 2 or edge_degree[1]==1:
                    end_nodes.append(edge_degree[0])

        #self.logger_main.debug("degree_list: %s" %(degree_list))
        #self.logger_main.debug("end nodes list:%s" %(end_nodes))
        #for n in range(2, len(monitor_candidate_list) + 1, 1):
        #for n in range(2, 3, 1):
        monitors=[]
        monitors_deployment_percentage=[]
        rate_of_optimal_actions_list_with_increasing_monitors=[]
        path_oscilation_list_with_increasing_monitors=[]
        traffic_overhead_every_200_iterations_with_increasing_monitors = []
        #end_nodes=[]
        for m_p in monitor_pert_list:
        #for m_p in [20, 30]:
            monitors_deployment_percentage.append(m_p)
            n = int((m_p / 100) * len(G.nodes))
            if n==2:
                n=3
            if n <= len(end_nodes):
                rest_end_nodes = [elem for elem in end_nodes if elem not in monitors]
                # self.logger_main.debug(f"rest node {rest_end_nodes}")
                select = sample(rest_end_nodes, k=n - len(monitors))
                # self.logger_main.debug(f"select {select}")
                monitors = monitors + select
            else:
                monitors = self.topo.deploy_monitor(G, n, end_nodes)
            self.logger_main.info("deloy %d pert monitors: %s" % (m_p, monitors))
            #trimedG=mynetwork.topo.trimNetwrok(G, monitors)
            trimedG = G
            nx.write_gml(trimedG, "%sGraph_%s_%s.gml" % (self.trimedGraph_Dir, type, str(m_p)))
         ##dfdfdf
            expo_count, total_mse, total_mse_optimal_edges_array, rewards_mse_list, optimal_delay, edge_exploration_during_training, average_computed_edge_num, optimal_path_selected_rate, avg_diff_of_delay_from_optimal,average_probing_links_origin, average_probing_links_reduced, rate_of_optimal_actions_list, path_oscilation_list, traffic_overhead_every_200_iterations = self.run_MAB(
                trimedG, monitors, llc_factor)
            monitors_list.append(monitors)
            explored_edges_rate.append(expo_count / len(trimedG.edges))
            total_edge_mse_list_with_increasing_monitors.append(total_mse)
            total_optimal_edges_mse_list_with_increasing_monitors.append(total_mse_optimal_edges_array)
            total_rewards_mse_list.append(rewards_mse_list)
            total_edge_exploration_during_training_list.append(edge_exploration_during_training)
            average_computed_edge_rate_during_training.append(average_computed_edge_num / len(trimedG.edges))
            optimal_path_selected_percentage_list.append(optimal_path_selected_rate)
            avg_diff_of_delay_from_optimal_list.append(avg_diff_of_delay_from_optimal)
            average_probing_links_origin_list.append(average_probing_links_origin)
            average_probing_links_reduced_list.append(average_probing_links_reduced)
            rate_of_optimal_actions_list_with_increasing_monitors.append(rate_of_optimal_actions_list)
            path_oscilation_list_with_increasing_monitors.append(path_oscilation_list)
            traffic_overhead_every_200_iterations_with_increasing_monitors.append(traffic_overhead_every_200_iterations)
            self.logger_main.info("edges explored: %f" % (expo_count / len(trimedG.edges)))
            self.logger_main.info("edges computed: %f" % (average_computed_edge_num / len(trimedG.edges)))
            # np.savetxt("mse_with_NT_in_training_node%s.txt" %(len(G.nodes)), np_array_total_mse, delimiter=",")
            self.logger_main.info("percentage of the optimal path selected: %f" % (optimal_path_selected_rate))
            self.logger_main.info(" abs diff from the real optimal path: %f" %(avg_diff_of_delay_from_optimal))
            self.logger_main.info(" %s pert is done, current shape of rate_of_optimal_actions_list_with_increasing_monitors %s" %(m_p, np.array(rate_of_optimal_actions_list_with_increasing_monitors).shape))
            self.logger_main.info(" %s pert is done, current shape of rate_of_path_oscilatoion_with_increasing_monitors %s" %(m_p, np.array(path_oscilation_list_with_increasing_monitors).shape))
            self.topo.draw_edge_delay_sample(G, type, node_num, p)

        arr = np.array(average_computed_edge_rate_during_training)
        np.savetxt('%sidentificable edges rate with increasing monitors' % (self.directory), arr)
        np.savetxt(mynetwork.directory + 'links_delay_during_training_with_different_monitor_size.txt',
                   total_mse)
        # self.plotter.plot_rewards_mse_along_with_different_monitors(monitors_deployment_percentage,total_rewards_mse_list)
        # self.plotter.plot_bar_edge_exploration_training_with_increasing_monitor(monitors_deployment_percentage, explored_edges_rate)
        self.plotter.plot_edge_computed_rate_during_training(monitors_deployment_percentage, average_computed_edge_rate_during_training)
        return optimal_path_selected_percentage_list, avg_diff_of_delay_from_optimal_list, total_edge_mse_list_with_increasing_monitors, \
               total_optimal_edges_mse_list_with_increasing_monitors, monitors_deployment_percentage, average_probing_links_origin_list,\
               average_probing_links_reduced_list, rate_of_optimal_actions_list_with_increasing_monitors, \
               path_oscilation_list_with_increasing_monitors, traffic_overhead_every_200_iterations_with_increasing_monitors, \
               average_computed_edge_rate_during_training


    def plot_edge_computed_rate_bar_with_different_topology_size(self):
        self.plotter.plot_edge_computed_rate_with_different_topology_size()

    def plot_learning_error_of_total_edges(self):
        monitors_deployment_percentage=[10, 20, 30, 40, 50]
        total_edge_avg_mse_list_with_increasing_monitors=np.loadtxt("mse_results/new_avg.txt")
        total_edge_std=np.loadtxt("mse_results/new_std.txt")
        # total_edge_avg_mse_list_with_increasing_monitors_new=[]
        # total_edge_avg_mse_list_with_increasing_monitors_new.append(total_edge_avg_mse_list_with_increasing_monitors[0])
        # total_edge_avg_mse_list_with_increasing_monitors_new.append(total_edge_avg_mse_list_with_increasing_monitors[2])
        # total_edge_avg_mse_list_with_increasing_monitors_new.append(total_edge_avg_mse_list_with_increasing_monitors[1])
        # total_edge_avg_mse_list_with_increasing_monitors_new.append(total_edge_avg_mse_list_with_increasing_monitors[3])
        # pert_50=[i-1 for i in total_edge_avg_mse_list_with_increasing_monitors[4]]
        # total_edge_avg_mse_list_with_increasing_monitors_new.append(pert_50)
        # for i in np.arange(len(total_edge_std)):
        #     for j in np.arange(len(total_edge_std[0])):
        #         total_edge_std[i][j]=total_edge_std[i][j]/2
        # # std_err_new=[]
        # std_err_new.append(total_edge_std[0]/2)
        # std_err_new.append(total_edge_std[2]/2)
        # std_err_new.append(total_edge_std[1]/2)
        # std_err_new.append(total_edge_std[4]/2)
        # std_err_new.append(total_edge_std[3])

        self.plotter.plot_total_edge_delay_mse_with_increasing_monitor_training(monitors_deployment_percentage,
                                                                   total_edge_avg_mse_list_with_increasing_monitors,
                                                                   total_edge_std)

    def plot_final_result(self, mynetwork):
        #plot the scalability performance in Barabasi 50 with 30% monitors deployed but varies the network size
        monitors_deployment_percentage = [10,20,30,40,50]
        #myapproach_optimal_path_selected_rate = [0.7303425, 0.784525, 0.820643333, 0.838918553, 0.902852585]
        #myapproach_optimal_path_selected_rate=[0.6682375, 0.708409445 ,0.721732615, 0.751597365, 0.80618583]

        subito_op_rate = [0.709256667, 0.75291074, 0.787211903, 0.80995684, 0.869051943] #found
        subito_op_rate_std=[0.0887999 ,0.05018362, 0.01950961, 0.03722164, 0.03438654]
        UCB1_op_rate = [0.56466, 0.557230555, 0.56462738, 0.56482263, 0.57180575]
        UCB1_op_rate_std=[0.08692476, 0.06008045, 0.03377418, 0.01845609, 0.01156429]
        subito_perfect_op_rate = [0.8338175, 0.864745, 0.86949619, 0.87432776, 0.8759815] #found
        subito_perfect_op_rate_std=[0.08510271, 0.06135779, 0.01264417, 0.01889351, 0.01521417]
        BoundNT_op_rate=[0.48122,  0.42990222, 0.41871429, 0.39760421, 0.375116]
        BundNT_op_rate_std=[0.08372442, 0.00986752, 0.02805065, 0.01857001, 0.00657999]

        UCB1_diff = [4.55981845, 4.32474493, 4.284358305, 4.25667079, 4.15141293]
        UCB1_diff_std=[1.140594,  0.61546221, 0.33471532, 0.30259693, 0.17466537]
        subito_diff = [3.572746567, 2.762555307, 2.438297613, 2.284643933, 1.743578883] #found
        subito_diff_std=[0.83206654, 0.35860155, 0.13579816, 0.23556937, 0.20946304]
        subito_perfect_diff = [2.236884885, 1.71805823, 1.60399359, 1.53417783, 1.49493546]#found
        subito_perfect_diff_std=[0.80836009, 0.39592167, 0.13720557, 0.16393052, 0.1090322 ]
        BoundNT_diff=[5.38756999, 5.86999576, 6.06494927, 6.2990515, 6.444486]
        BoundNT_diff_std=[1.00782274, 0.58879442, 0.70106653, 0.41320291, 0.13556835]

        UCB1_traffic_overhead=[45.6643095,197.969513,459.798732,834.551768,1305.78879]
        UCB1_traffic_overhead_std=[ 4.17969365,  6.7075481,  12.45923412, 14.04451219, 13.89895365]
        #subito_MAB_trffic_overhead=[ 36.44358334,166.6565, 384.0626667,	700.6428667, 1107.435267]
        subito_NT_traffic_overhead=[31.25533334, 103.3804,	146.1748,	175.6659333, 190.4278667]
        subito_NT_traffic_overhead_std=[2.63268111, 8.30936683, 2.67356925, 7.18697759, 5.02490679]
        boundNT_traffic_overhead=[30.3286667,79.7326667,129.9286,178.825867,220.262267]
        boundNT_traffic_overhead_std=[2.87539421,  9.99963958, 10.00933721,  1.68800559,  3.9658315]
        #mynetwork.plotter.plot_percentage_of_optimal_path_selected_rate_BR_50nodes(monitors_deployment_percentage, subito_op_rate, UCB1_op_rate, subito_perfect_op_rate)
        #mynetwork.plotter.plot_abs_delay_of_optimal_path_selected_from_mean_BR_50nodes(monitors_deployment_percentage, subito_diff, UCB1_diff, subito_perfect_diff)
        mynetwork.plotter.plot_percentage_of_optimal_path_selected_rate_line(monitors_deployment_percentage, subito_op_rate,subito_op_rate_std, UCB1_op_rate,UCB1_op_rate_std, subito_perfect_op_rate, subito_perfect_op_rate_std, BoundNT_op_rate, BundNT_op_rate_std, "BR50")
        mynetwork.plotter.plot_abs_delay_of_optimal_path_selected_from_mean_line(monitors_deployment_percentage, subito_diff, subito_diff_std,  UCB1_diff, UCB1_diff_std, subito_perfect_diff, subito_perfect_diff_std, BoundNT_diff, BoundNT_diff_std, "BR50")
        # mynetwork.plotter.plot_traffic_overhead_monitor_size(monitors_deployment_percentage, subito_NT_traffic_overhead, subito_NT_traffic_overhead_std, boundNT_traffic_overhead,boundNT_traffic_overhead_std, UCB1_traffic_overhead, UCB1_traffic_overhead_std, "BR50")
        mynetwork.plot_path_oscillation_BR50()

        #plot the scalability performance of network size 20, 40, 60, 80 nodes with fixed 30% monitors deployed
        topology_size=[20, 40, 60, 80]
        subito_op_rate = [0.86708, 0.78523, 0.729153167, 0.666417933]
        subito_op_rate_std=[0.11427922,0.05693109,0.0570062,0.03241127]
        UCB1_op_rate = [0.50193667, 0.44657879, 0.34701209, 0.3847663]
        UCB1_op_rate_std=[0.11036267,0.07090994,0.02299693, 0.02901814]
        subito_perfect_op_rate = [0.88412667, 0.85507955, 0.77260229, 0.74318261]  #found
        subito_perfect_op_rate_std=[0.05362887, 0.02650313,0.05879128, 0.0176218]
        boundNT_op_rate=[0.47973333, 0.4293, 0.37392418, 0.34462681]
        boundNT_op_rate_std=[0.08476537, 0.0438621, 0.05030017, 0.01102395 ]

        UCB1_diff = [4.30608774, 6.11897566, 6.15564568, 8.3113577]
        UCB1_diff_std=[0.90078678, 0.42840205,0.15996378, 0.46991514]
        subito_diff = [1.45996846, 2.430924276, 2.498190838, 3.326673313]
        subito_diff_std=[1.09429601,0.57149775,0.36118806,0.20588522]
        subito_perfect_diff = [1.29733573, 1.9665792, 2.29629007, 2.98189548] #found
        subito_perfect_diff_std=[0.64604337,0.26441096,0.36247737,0.13734982]
        boundNT_diff=[4.73952412, 5.84895738, 6.52524591, 7.91529036]
        boundNT_diff_std=[0.83719921, 0.36482026,0.55897507,0.34965089 ]
        #subito_MAB_trffic_overhead=[48.646, 255.37633333, 527.29533333, 1093.01733333]
        #subito_NT_traffic_overhead=[26.62933333, 107.21466667, 138.905, 258.91733333]
        #UCB1_traffic_overhead=[46.46057372, 271.77565043, 665.71047365, 1454.89766511]
        #boundNT_traffic_overhead=[25.5, 86, 129, 232]

        subito_NT_traffic_overhead = [0.2662933333, 1.0721466667, 1.38905, 2.5891733333]
        subito_NT_traffic_overhead_std=[0.0390979135,0.1244135817,0.1075188437,0.0998876226]
        UCB1_traffic_overhead = [0.4646057372, 2.7177565043, 6.6571047365, 14.5489766511]
        UCB1_traffic_overhead_std=[0.027943447,0.0742583262, 0.1160141243, 0.3466985383]
        boundNT_traffic_overhead = [0.25, 0.86, 1.29, 2.32]
        boundNT_traffic_overhead_std=[0.0400104998, 0.0675384551, 0.0920122094,0.0951357184]
        #mynetwork.plotter.plot_percentage_of_optimal_path_selected_rate_for_various_network_size_line(topology_size,subito_op_rate,UCB1_op_rate,subito_perfect_op_rate,boundNT_op_rate)
        # mynetwork.plotter.plot_abs_delay_of_optimal_path_selected_for_various_network_size_line(topology_size, subito_diff, subito_diff_std, UCB1_diff, UCB1_diff_std, subito_perfect_diff,subito_perfect_diff_std, boundNT_diff, boundNT_diff_std)
        # mynetwork.plotter.plot_traffic_overhead_network_size(subito_NT_traffic_overhead, subito_NT_traffic_overhead_std, boundNT_traffic_overhead,boundNT_traffic_overhead_std,UCB1_traffic_overhead, UCB1_traffic_overhead_std, "size-20-80")

        #plot the experiments for real infrastructure
        monitors_deployment_percentage = [10, 20, 30, 40, 50]
        myapproach_optimal_path_selected_rate = [0.8730325,	0.865852223, 0.84571643, 0.835979885,0.843303463]
        UCB1_op_rate = [0.67076,	0.68020833,	0.67219667,	0.66329536,	0.67381338]
        #UCB1_op_rate_std=[0.1130971,  0.04305576, 0.08356142, 0.03220476, 0.01342114]
        UCB1_op_rate_std=[0.0630971,  0.04305576, 0.08356142, 0.03220476, 0.01342114]
        subito_perfect_op_rate=[0.876075,	0.9192025,	0.911354763,	0.908002727,	0.91525641]
        #subito_perfect_op_rate_std=[0.09269107, 0.07641893, 0.05654337, 0.05471862, 0.04752676]
        subito_perfect_op_rate_std=[0.023475945,	0.008623167, 0.016431394, 0.026052381,0.001279397]
        subito_op_rate=[0.84235, 0.874365,	0.859673215, 0.88029, 0.887951095]
        #subito_op_rate_std=[0.11074467, 0.13180377, 0.04078737, 0.03164284, 0.04411764]
        subito_op_rate_std=[0.0256062,	0.018923426,	0.013676243,	0.013538655,	0.008974321]
        BoundNT_op_rate=[0.72553333, 0.68322, 0.60758571, 0.56908, 0.54016923]
        BoundNT_op_rate_std=[0.17592498, 0.12137273, 0.02560268, 0.05189104, 0.00202802]

        UCB1_diff = [3.92730099,	3.55005621,	3.76985005,	4.01648821,	4.05722965]
        UCB1_diff_std=[1.03892507, 0.61495123, 0.62701585, 0.40280497, 0.14067127]
        subito_perfect_diff= [1.704276463, 1.125456645, 1.099625047, 1.085538627, 1.023573517]
        subito_perfect_diff_std = [1.3611675, 0.72666219, 0.26666809, 0.86340987, 0.6518846]
        subito_perfect_diff_std=[0.170975096,	0.110598572,	0.079300857,	0.111459991,	0.033213449]
        #subito_perfect_diff_std=[0.202118854,	0.185842434,	0.079300857,	0.111459991,	0.039566863]
        subito_diff=[2.266471527, 1.8255659,	1.984103853, 1.734049937, 1.540486353]
        subito_diff_std=[0.66163648, 0.60636749, 0.3768265,  0.4255334, 0.43353286]
        BoundNT_diff=[2.69067996, 3.66938621, 4.9628864,5.71129116, 6.07322507]
        BoundNT_diff_std=[0.22138409,  0.139108262, 0.15452393, 0.5495862,  0.04428794]

        subito_traffic_overhead = [11.5994667, 26.8632667, 52.7939333,67.7846,70.6566667]
        subito_traffic_overhead_std=[2.76023858, 4.26181478, 6.11365698, 8.30881461, 5.16263667]
        #boundNT_traffic_overhead= [8.822599998, 24.58526667,49.768, 75.25953334,98.9822]
        boundNT_traffic_overhead=[12.0218, 26.365, 59.85253333, 80.51173333, 105.87053333]
        boundNT_traffic_overhead_std=[4.18675088, 7.39921935, 3.46870565, 6.53340539, 0.14215788]
        UCB1_traffic_overhead=[12.8108072, 45.8701801, 138.502935, 270.016278, 441.789193]
        #UCB1_traffic_overhead_std=[3.55635579, 3.59812508, 8.17196001, 9.63812801, 3.0742781]
        UCB1_traffic_overhead_std=[4.82431952, 9.10856858, 18.01807906, 10.47439213,  2.60035666]

        mynetwork.plotter.plot_percentage_of_optimal_path_selected_rate_line(monitors_deployment_percentage,subito_op_rate,subito_op_rate_std, UCB1_op_rate, UCB1_op_rate_std, subito_perfect_op_rate, subito_perfect_op_rate_std,BoundNT_op_rate, BoundNT_op_rate_std,"BTN")
        mynetwork.plotter.plot_abs_delay_of_optimal_path_selected_from_mean_line(monitors_deployment_percentage, subito_diff, subito_diff_std, UCB1_diff, UCB1_diff_std, subito_perfect_diff,subito_perfect_diff_std, BoundNT_diff, BoundNT_diff_std,"BTN")
        #mynetwork.plotter.plot_traffic_overhead_monitor_size(monitors_deployment_percentage,subito_traffic_overhead,subito_traffic_overhead_std, boundNT_traffic_overhead, boundNT_traffic_overhead_std, UCB1_traffic_overhead, UCB1_traffic_overhead_std, "BTN")
        mynetwork.plot_path_oscillation_BTN()

        # plot the experiments for real infrastructure with background traffic dataset
        monitors_deployment_percentage = [60, 70, 80, 90, 100]
        UCB1_op_rate = [0.75854667, 0.78922857, 0.82021429, 0.84372222, 0.87918545]
        UCB1_op_rate_std = [0.03251421, 0.02328221, 0.03952764, 0.02224307, 0.01842566]
        subito_perfect_op_rate = [0.99933667, 0.99993571, 0.99994643, 0.99995139, 1]
        subito_perfect_op_rate_std = [0.00183745, 0.00014465, 0.00016111, 0.00021189, 0]
        subito_op_rate = [0.98549333, 0.98960952, 0.9999375 , 0.99645278, 1]
        subito_op_rate_std = [0.02895175, 0.02543533, 0.00027243, 0.01543651, 0]
        BoundNT_op_rate = [0.73063333, 0.80212857, 0.84673929, 0.89622222, 0.9314]
        BoundNT_op_rate_std = [3.02247139e-03, 1.33064193e-02,2.38270878e-02, 2.47686674e-02, 1.11022302e-16]


        UCB1_diff = [5.97670558, 5.12548035, 4.33209074,  3.57787085, 2.6660592]
        UCB1_diff_std = [0.95178187,0.59634358,0.74085781,0.40131891, 0.41262554]
        subito_perfect_diff = [0.00889449, 0.0012855,0.000909, 0.00102495, 0.00023081]
        subito_perfect_diff_std = [0.02080813, 0.00214353, 0.00150399, 0.00407369, 0.00038691]
        subito_diff =[0.22581931,0.13732406, 0.00192705, 0.03948959, 0.00028845]
        subito_diff_std =[5.13821589e-01, 3.62242803e-01, 7.79013518e-03, 1.69989221e-01,4.54388561e-04]
        BoundNT_diff = [7.16418305, 4.53047573, 3.25391783,  1.86352191, 1.08432175]
        BoundNT_diff_std = [0.15042113, 0.39031218, 0.7020151,  0.50369672, 0]


        subito_traffic_overhead = [14.01314286, 14.12528571, 11.806, 17.4, 14]
        subito_traffic_overhead_std =[3.27442114, 5.13104993, 2.2249818, 2.57681975, 0]
        boundNT_traffic_overhead = [24.70307143, 30.95264286, 31.48592857, 32.37321429, 30.91307143]
        boundNT_traffic_overhead_std = [0.19270021, 1.34264828, 1.57803152, 2.00080227, 0.07279413]
        UCB1_traffic_overhead = [ 44.12532189, 57.29327611, 71.15221745, 88.30715308, 124.91359084]
        UCB1_traffic_overhead_std = [0.49701996, 0.40116245, 1.60501788, 1.52586872, 1.04855566]

        # mynetwork.plotter.plot_percentage_of_optimal_path_selected_rate_line(monitors_deployment_percentage,subito_op_rate,subito_op_rate_std, UCB1_op_rate, UCB1_op_rate_std, subito_perfect_op_rate, subito_perfect_op_rate_std,BoundNT_op_rate, BoundNT_op_rate_std,"NSF")
        # mynetwork.plotter.plot_abs_delay_of_optimal_path_selected_from_mean_line(monitors_deployment_percentage, subito_diff, subito_diff_std, UCB1_diff, UCB1_diff_std, subito_perfect_diff,subito_perfect_diff_std, BoundNT_diff, BoundNT_diff_std,"NSF")
        # mynetwork.plotter.plot_traffic_overhead_monitor_size(monitors_deployment_percentage,subito_traffic_overhead,subito_traffic_overhead_std, boundNT_traffic_overhead, boundNT_traffic_overhead_std, UCB1_traffic_overhead, UCB1_traffic_overhead_std, "NSF")
        # mynetwork.plot_path_oscillation_NSF(monitors_deployment_percentage)

        edge_compute_rate=[0.54948571, 0.78311429, 0.85048571, 0.96268571, 1]
        edge_compute_rate_std=[0.00557223, 0.20122672, 0.07475733, 0.07462857, 0.  ]
        # mynetwork.plotter.plot_edge_compute_rate_subito(monitors_deployment_percentage, edge_compute_rate, edge_compute_rate_std)
        # mynetwork.plot_learning_error_of_total_edges_NSF(monitors_deployment_percentage)

        # plot the experiments for real infrastructure with background traffic dataset + real traffic trails
        monitors_deployment_percentage = [60, 70, 80, 90, 100]
        UCB1_op_rate = [0.75854667, 0.78922857, 0.82021429, 0.84372222, 0.87918545]
        UCB1_op_rate_std = [0.03251421, 0.02328221, 0.03952764, 0.02224307, 0.01842566]
        subito_perfect_op_rate = [0.99933667, 0.99993571, 0.99994643, 0.99995139, 1]
        subito_perfect_op_rate_std = [0.00183745, 0.00014465, 0.00016111, 0.00021189, 0]
        subito_op_rate = [0.98549333, 0.98960952, 0.9999375, 0.99645278, 1]
        subito_op_rate_std = [0.02895175, 0.02543533, 0.00027243, 0.01543651, 0]
        BoundNT_op_rate = [0.73063333, 0.80212857, 0.84673929, 0.89622222, 0.9314]
        BoundNT_op_rate_std = [3.02247139e-03, 1.33064193e-02, 2.38270878e-02, 2.47686674e-02, 1.11022302e-16]

        UCB1_diff = [5.97670558, 5.12548035, 4.33209074, 3.57787085, 2.6660592]
        UCB1_diff_std = [0.95178187, 0.59634358, 0.74085781, 0.40131891, 0.41262554]
        subito_perfect_diff = [0.00889449, 0.0012855, 0.000909, 0.00102495, 0.00023081]
        subito_perfect_diff_std = [0.02080813, 0.00214353, 0.00150399, 0.00407369, 0.00038691]
        subito_diff = [0.22581931, 0.13732406, 0.00192705, 0.03948959, 0.00028845]
        subito_diff_std = [5.13821589e-01, 3.62242803e-01, 7.79013518e-03, 1.69989221e-01, 4.54388561e-04]
        BoundNT_diff = [7.16418305, 4.53047573, 3.25391783, 1.86352191, 1.08432175]
        BoundNT_diff_std = [0.15042113, 0.39031218, 0.7020151, 0.50369672, 0]

        subito_traffic_overhead = [30, 24.6, 20.2, 20.8, 14] #updated
        subito_traffic_overhead_std = [0, 2.8, 1.6, 2.4, 0.] #updated
        boundNT_traffic_overhead = [24.70307143, 30.95264286, 31.48592857, 32.37321429, 30.91307143]
        boundNT_traffic_overhead_std = [0.19270021, 1.34264828, 1.57803152, 2.00080227, 0.07279413]
        UCB1_traffic_overhead = [44.12532189, 57.29327611, 71.15221745, 88.30715308, 124.91359084]
        UCB1_traffic_overhead_std = [0.49701996, 0.40116245, 1.60501788, 1.52586872, 1.04855566]

        # mynetwork.plotter.plot_percentage_of_optimal_path_selected_rate_line(monitors_deployment_percentage,subito_op_rate,subito_op_rate_std, UCB1_op_rate, UCB1_op_rate_std, subito_perfect_op_rate, subito_perfect_op_rate_std,BoundNT_op_rate, BoundNT_op_rate_std,"NSF")
        # mynetwork.plotter.plot_abs_delay_of_optimal_path_selected_from_mean_line(monitors_deployment_percentage, subito_diff, subito_diff_std, UCB1_diff, UCB1_diff_std, subito_perfect_diff,subito_perfect_diff_std, BoundNT_diff, BoundNT_diff_std,"NSF")
        # mynetwork.plotter.plot_traffic_overhead_monitor_size(monitors_deployment_percentage,subito_traffic_overhead,subito_traffic_overhead_std, boundNT_traffic_overhead, boundNT_traffic_overhead_std, UCB1_traffic_overhead, UCB1_traffic_overhead_std, "NSF")
        # mynetwork.plot_path_oscillation_NSF(monitors_deployment_percentage)

        edge_compute_rate = [0.54755513, 0.74165868, 0.83216683, 0.95800575, 1. ] #updaed
        edge_compute_rate_std = [0.00011743, 0.09697993, 0.08391678, 0.08398849, 0 ] #updated
        mynetwork.plotter.plot_edge_compute_rate_subito(monitors_deployment_percentage, edge_compute_rate, edge_compute_rate_std)
        #mynetwork.plot_learning_error_of_total_edges_NSF(monitors_deployment_percentage)
    def plot_traffic_overhead_of_subito(self):
        multi_times_avg_traffic_overhead_every_200_iterations_with_increasing_monitors=[]
        multi_times_avg_traffic_overhead_every_200_iterations_with_increasing_monitors.append([23.88, 23.7225, 23.725, 23.6975, 23.799, 23.781666666666666, 23.826428571428572, 23.844375, 23.850555555555555, 23.8665, 23.89727272727273, 23.855, 23.860384615384614, 23.84464285714286, 23.845666666666666])
        multi_times_avg_traffic_overhead_every_200_iterations_with_increasing_monitors.append([85.345, 86.1, 86.45666666666666, 86.28125, 85.929, 85.51833333333333, 85.405, 85.40875, 85.46888888888888, 85.375, 85.31090909090909, 85.35958333333333, 85.425, 85.49035714285715, 85.40033333333334])
        multi_times_avg_traffic_overhead_every_200_iterations_with_increasing_monitors.append([143.095, 141.91, 141.45833333333334, 141.43375, 141.331, 141.455, 141.49642857142857, 141.3525, 141.38222222222223, 141.269, 141.20272727272726, 141.28625, 141.28192307692308, 141.185, 141.30733333333333])
        multi_times_avg_traffic_overhead_every_200_iterations_with_increasing_monitors.append([199.185, 196.55, 196.16333333333333, 195.39875, 195.389, 195.3325, 195.67785714285714, 195.510625, 195.2588888888889, 195.1845, 195.3318181818182, 195.22666666666666, 195.1380769230769, 194.9882142857143, 194.85866666666666])
        multi_times_avg_traffic_overhead_every_200_iterations_with_increasing_monitors.append([222.245, 223.7175, 223.51, 223.69875, 223.841, 223.6375, 223.72, 223.84875, 223.62722222222223, 223.5515, 223.49318181818182, 223.49041666666668, 223.44307692307692, 223.33714285714285, 223.45566666666667])
        self.plotter.plot_avg_traffic_overhead_every_200_iterations([10,20,30,40,50], multi_times_avg_traffic_overhead_every_200_iterations_with_increasing_monitors)


    def plot_path_oscillation_BR50(self):
        monitor_pert=[10,20,30,40,50]
        BoundNT=np.loadtxt("path_oscillation_BR50/BoundNT_path_oscillation_BR50_10%-50%.txt")
        BoundNT_std=np.loadtxt("path_oscillation_BR50/path_osc_boundNT_std_new.txt")
        Subito=np.loadtxt("path_oscillation_BR50/Subito_ocsillition_every_200_times_BR50_10%-50%.txt")
        Subito_std=np.loadtxt("path_oscillation_BR50/path_osc_subito_std_new.txt")
        UCB1=np.loadtxt("path_oscillation_BR50/UBC1_ocsillition_every_200_times_BR50_10%-50%_baseline.txt")
        UCB1_std=np.loadtxt("path_oscillation_BR50/path_osc_UBC1_std_new.txt")

        self.plotter.plot_avg_path_oscilation_every_200_times_withname(monitor_pert,Subito, Subito_std,"Subito_path_oscillation_BR50")
        self.plotter.plot_avg_path_oscilation_every_200_times_withname(monitor_pert,BoundNT, BoundNT_std, "BoundNT_path_oscillation_BR50")
        self.plotter.plot_avg_path_oscilation_every_200_times_withname(monitor_pert,UCB1, UCB1_std, "UCB1_path_oscillation_BR50")

    def plot_path_oscillation_BTN(self):
        monitor_pert = [10, 20, 30, 40, 50]
        BoundNT = np.loadtxt("path_oscillation_BTN/BoundNT_Ocsillation.txt")
        BoundNT_std=np.loadtxt("path_oscillation_BTN/path_osc_boundNT_std.txt")
        Subito = np.loadtxt("path_oscillation_BTN/Subito_ocsillation_BTN.txt")
        Subito_std=np.loadtxt("path_oscillation_BTN/subito_path_ocsillation_std_BTN.txt")
        UCB1 = np.loadtxt("path_oscillation_BTN/UCB1_oscillation_BTN.txt")
        UCB1_std=np.loadtxt("path_oscillation_BTN/new_std_UCB1.txt")
        self.plotter.plot_avg_path_oscilation_every_200_times_withname(monitor_pert, Subito, Subito_std,
                                                                       "Subito_path_oscillation_BTN")
        self.plotter.plot_avg_path_oscilation_every_200_times_withname(monitor_pert, BoundNT, BoundNT_std,
                                                                       "BoundNT_path_oscillation_BTN")
        self.plotter.plot_avg_path_oscilation_every_200_times_withname(monitor_pert, UCB1, UCB1_std,"UCB1_path_oscillation_BTN")

    def plot_path_oscillation_NSF(self, monitor_pert):
        BoundNT = np.loadtxt("path_oscillation_NSF/path_osc_avg_boundNT.txt")
        BoundNT_std = np.loadtxt("path_oscillation_NSF/path_osc_std_boundNT.txt")
        Subito = np.loadtxt("path_oscillation_NSF/path_osc_avg_subito.txt")
        Subito_std = np.loadtxt("path_oscillation_NSF/path_osc_std_subito.txt")
        UCB1 = np.loadtxt("path_oscillation_NSF/path_osc_avg_UCB1.txt")
        UCB1_std = np.loadtxt("path_oscillation_NSF/path_osc_std_UCB1.txt")
        self.plotter.plot_avg_path_oscilation_every_200_times_withname(monitor_pert, Subito, Subito_std,
                                                                       "Subito_path_oscillation_NSF")
        self.plotter.plot_avg_path_oscilation_every_200_times_withname(monitor_pert, BoundNT, BoundNT_std,
                                                                       "BoundNT_path_oscillation_NSF")
        self.plotter.plot_avg_path_oscilation_every_200_times_withname(monitor_pert, UCB1, UCB1_std,
                                                                       "UCB1_path_oscillation_NSF")

    def calculate_multi_times_serirs_results_avg_std(self, monitors_deployment_percentage, multi_times_total, n, file_dir, filename):
        # calculate the average of mse of the total link delays and the std
        round_monitor_pert = len(monitors_deployment_percentage)
        multi_times_avg_mse= []
        multi_times_std_mse = []
        print(f"multi_times_total shape: {multi_times_total.shape}")
        for i in np.arange(round_monitor_pert):
            multi_times_arr_one_percentage = np.array([multi_times_total[i]])
            for j in np.arange(1, n):
                new_row=multi_times_total[i + round_monitor_pert * j]
                multi_times_arr_one_percentage=np.append(multi_times_arr_one_percentage,[new_row],axis=0)
            multi_times_avg_mse.append(np.average(multi_times_arr_one_percentage, axis=0))
            multi_times_std_mse.append(np.std(multi_times_arr_one_percentage, axis=0))
        np.savetxt(file_dir+filename+'_avg.txt', multi_times_avg_mse )
        np.savetxt(file_dir+filename+'_std.txt', multi_times_std_mse)

        '''
        for i in np.arange(round_monitor_pert):
            if i == 0:
                multi_times_arr_one_percentage = np.array([multi_times_total[i]])
            else:
                for j in np.arange(n):
                    new_row=multi_times_total[i + round_monitor_pert * j]
                    multi_times_arr_one_percentage=np.append(multi_times_arr_one_percentage,[new_row],axis=0)
            multi_times_avg_mse.append(np.average(multi_times_arr_one_percentage, axis=0))
            multi_times_std_mse.append(np.std(multi_times_arr_one_percentage, axis=0))
            np.savetxt(file_dir+filename+'_avg.txt', multi_times_avg_mse )
            np.savetxt(file_dir+filename+'_std.txt', multi_times_std_mse)
        '''
        return multi_times_avg_mse, multi_times_std_mse

    def plot_learning_error_of_total_edges_NSF(self, monitor_pert_list):
        monitors_deployment_percentage = monitor_pert_list
        total_edge_avg_mse_list_with_increasing_monitors = np.loadtxt(self.directory+"total_edge_mse/total_mse_error_avg.txt")
        total_edge_std = np.loadtxt(self.directory+"total_edge_mse/total_mse_error_std.txt")
        # total_edge_avg_mse_list_with_increasing_monitors = np.loadtxt(
        #     "mse_results_NSF/total_edge_mse/total_mse_error_avg.txt")
        # total_edge_std = np.loadtxt("mse_results_NSF/total_edge_mse/total_mse_error_std.txt")

        #total_edge_avg_mse_list_with_increasing_monitors = np.loadtxt(
        #    "./temp/5times/total_edge_mse/total_mse_error_avg.txt")
        #total_edge_std = np.loadtxt("./temp/5times/total_edge_mse/total_mse_error_std.txt")
        self.plotter.plot_total_edge_delay_mse_with_increasing_monitor_training(monitors_deployment_percentage,
                                                                                total_edge_avg_mse_list_with_increasing_monitors,
                                                                                total_edge_std, "total")
    #todo
    def plot_learning_error_of_total_opt_edges_NSF(self, monitor_pert_list):
        monitors_deployment_percentage = monitor_pert_list
        total_edge_avg_mse_list_with_increasing_monitors = np.loadtxt(self.directory+"total_opt_edge_mse/total_opt_mse_error_avg.txt")
        total_edge_std = np.loadtxt(self.directory+"total_opt_edge_mse/total_opt_mse_error_std.txt")
        #total_edge_avg_mse_list_with_increasing_monitors = np.loadtxt(
        #    "./temp/5times/total_edge_mse/total_mse_error_avg.txt")
        #total_edge_std = np.loadtxt("./temp/5times/total_edge_mse/total_mse_error_std.txt")
        self.plotter.plot_total_edge_delay_mse_with_increasing_monitor_training(monitors_deployment_percentage,
                                                                                total_edge_avg_mse_list_with_increasing_monitors,
                                                                                total_edge_std,"opt" )

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


mynetwork=main(298)
mynetwork.plot_final_result(mynetwork)
G =mynetwork.creat_topology(topo_type, num_node, degree)
#mynetwork.plot_path_oscillation_BTN()
path_osc_dir=mynetwork.directory+'path_oscillation/'
os.mkdir(path_osc_dir)
path_total_mse_dir=mynetwork.directory+'total_edge_mse/'
os.mkdir(path_total_mse_dir)
path_total_opt_mse_dir=mynetwork.directory+'total_opt_edge_mse/'
os.mkdir(path_total_opt_mse_dir)
path_total_computed_edge_dir=mynetwork.directory+'total_computed_edge/'
os.mkdir(path_total_computed_edge_dir)
file_list_total_mse=[]
file_list_osc_dir=[]
file_list_compute_edge=[]
monitor_pert_list=[60, 70, 80, 90, 100]
while(i<n):
    #mynetwork=main(3000)
    #G =mynetwork.creat_topology(topo_type, num_node, degree)
    #mynetwork.tomography_verification(G,'weight')   #here the assigned delay should be 1, place modify the topo.assign_link_delay() function
    optimal_path_selected_percentage_list, avg_diff_of_delay_from_optimal_list,total_edge_mse_list_with_increasing_monitors,\
    total_optimal_edges_mse_list_with_increasing_monitors,monitors_deployment_percentage, average_probing_links_origin_list, \
    average_probing_links_reduced_list, rate_of_optimal_actions_list_with_increasing_monitors, path_oscilation_list_with_increasing_monitors, \
    traffic_overhead_every_200_iterations_with_increasing_monitors, average_computed_edge_rate_during_training = \
    mynetwork.MAB_with_increasing_monitors(G,topo_type,len(G.nodes),degree, llc_factor, monitor_pert_list)

    #print("n=%d" %(i))
    #print(optimal_path_selected_percentage_list,avg_diff_of_delay_from_optimal_list)
    # save the 2D array data to a txt file
    np.savetxt(path_total_mse_dir+'%s.txt' %(i), total_edge_mse_list_with_increasing_monitors)
    np.savetxt(path_total_opt_mse_dir+ '%s.txt' %(i), total_optimal_edges_mse_list_with_increasing_monitors)
    np.savetxt(path_osc_dir + '%s.txt' %(i), path_oscilation_list_with_increasing_monitors)
    np.savetxt(path_total_computed_edge_dir + '%s.txt' %(i), average_computed_edge_rate_during_training )
    file_list_total_mse.append('%s.txt' %(i))
    file_list_osc_dir.append('%s.txt' %(i))
    file_list_compute_edge.append('%s.txt' %(i))
    if i==0:
        multi_times_mse_total_link_delay_array=np.array(total_edge_mse_list_with_increasing_monitors, dtype=float)
        multi_times_mse_opt_link_delay_array=np.array(total_optimal_edges_mse_list_with_increasing_monitors, dtype=float)
        multi_times_optimal_path_selected_percentage_array=np.array([optimal_path_selected_percentage_list])
        multi_times_avg_diff_of_delay_from_optimal_array=np.array([avg_diff_of_delay_from_optimal_list])
        multi_times_avg_n_probing_links_origin_array=np.array([average_probing_links_origin_list])
        multi_times_avg_n_probing_links_reduced_array=np.array([average_probing_links_reduced_list])
        #multi_times_rate_of_optimal_actions_list_with_increasing_monitors=np.array(rate_of_optimal_actions_list_with_increasing_monitors)
        multi_times_path_ocilations_with_increasing_monitors=np.array(path_oscilation_list_with_increasing_monitors)
        multi_times_compute_edge_with_increasing_monitors=np.array([average_computed_edge_rate_during_training])
        multi_times_traffic_overhead_every_200_iterations_with_increasing_monitors = np.array(traffic_overhead_every_200_iterations_with_increasing_monitors)
    else:
        current_mse_arrary=np.array(total_edge_mse_list_with_increasing_monitors)
        current_opt_mse_arrary=np.array(total_optimal_edges_mse_list_with_increasing_monitors)
        multi_times_mse_total_link_delay_array = np.concatenate((multi_times_mse_total_link_delay_array, current_mse_arrary), axis=0)
        multi_times_mse_opt_link_delay_array=np.concatenate((multi_times_mse_opt_link_delay_array, current_opt_mse_arrary), axis=0)
        multi_times_avg_diff_of_delay_from_optimal_array=np.append(multi_times_avg_diff_of_delay_from_optimal_array,np.array([avg_diff_of_delay_from_optimal_list]), axis=0)
        multi_times_avg_n_probing_links_origin_array=np.append(multi_times_avg_n_probing_links_origin_array,np.array([average_probing_links_origin_list]), axis=0)
        multi_times_avg_n_probing_links_reduced_array=np.append(multi_times_avg_n_probing_links_reduced_array, np.array([average_probing_links_reduced_list]), axis=0)
        multi_times_path_ocilations_with_increasing_monitors = np.concatenate((multi_times_path_ocilations_with_increasing_monitors,np.array(path_oscilation_list_with_increasing_monitors)), axis=0)
        #current_rate_of_optimal_actions_list_with_increasing_monitor_size = np.array(rate_of_optimal_actions_list_with_increasing_monitors)
        multi_times_optimal_path_selected_percentage_array=np.append(multi_times_optimal_path_selected_percentage_array, [np.array(optimal_path_selected_percentage_list)], axis=0)
        multi_times_traffic_overhead_every_200_iterations_with_increasing_monitors = np.add(multi_times_traffic_overhead_every_200_iterations_with_increasing_monitors, np.array(traffic_overhead_every_200_iterations_with_increasing_monitors))
        multi_times_compute_edge_with_increasing_monitors=np.append(multi_times_compute_edge_with_increasing_monitors,[np.array(average_computed_edge_rate_during_training)], axis=0)
    np.savetxt(mynetwork.directory+'optimal_actions.txt', multi_times_optimal_path_selected_percentage_array)
    np.savetxt(mynetwork.directory+'avg_regret.txt', multi_times_avg_diff_of_delay_from_optimal_array)
    np.savetxt(mynetwork.directory+'monitoring_overhead.txt', multi_times_avg_n_probing_links_reduced_array)
    np.savetxt(mynetwork.directory+'optimal_actions.txt', multi_times_optimal_path_selected_percentage_array )
    i += 1

#multi_times_avg_traffic_overhead_every_200_iterations_with_increasing_monitors=multi_times_traffic_overhead_every_200_iterations_with_increasing_monitors/n
#np.savetxt(mynetwork.directory + 'links_delay_during_training_with_different_monitor_size_total.txt', multi_times_avg_mse_total_link_delay_array)
#np.savetxt(mynetwork.directory + 'optimal_links_delay_during_training_with_different_monitor_size_total.txt', multi_times_avg_mse_total_optimal_links_delay_array)
mynetwork.logger_main.info(f"monitor deployment pert list: {monitor_pert_list}")
mynetwork.logger_main.info("Statistics:")
mynetwork.logger_main.info("Before average: percentage of the optimal path selected:")
mynetwork.logger_main.info(multi_times_optimal_path_selected_percentage_array)
mynetwork.logger_main.info("Before average: diff from the real optimal path: ")
mynetwork.logger_main.info(multi_times_avg_diff_of_delay_from_optimal_array)
mynetwork.logger_main.info("Before average: average probing links in reduced selected path:")
mynetwork.logger_main.info(multi_times_avg_n_probing_links_reduced_array)
mynetwork.logger_main.info("Before average: # of computed edge:")
mynetwork.logger_main.info(multi_times_compute_edge_with_increasing_monitors)

#mynetwork.logger_main.info("Before average: rate of the optimal actions shape: ")
#mynetwork.logger_main.info(multi_times_rate_of_optimal_actions_list_with_increasing_monitors.shape)

#AVG.mse of total edge delays over 3000 times, stored in the file
multi_times_avg_mse_total_link_delay_array,mutil_time_std_mse_total_links_delay_array =mynetwork.calculate_multi_times_serirs_results_avg_std(monitors_deployment_percentage, multi_times_mse_total_link_delay_array, n, path_total_mse_dir, 'total_mse_error')

##AVG.mse of total opt edge delays over 3000 times, stored in the file
multi_times_avg_opt_mse_total_link_delay_array,mutil_time_std_opt_mse_total_links_delay_array =mynetwork.calculate_multi_times_serirs_results_avg_std(monitors_deployment_percentage, multi_times_mse_opt_link_delay_array, n, path_total_opt_mse_dir, 'total_opt_mse_error')

#AVG.regret - statistic of avg and std of the regret
multi_avg_percentage_of_abs_diff_from_optimal=np.average(multi_times_avg_diff_of_delay_from_optimal_array,axis=0)
multi_std_percentage_of_abs_diff_from_optimal=np.std(multi_times_avg_diff_of_delay_from_optimal_array, axis=0)
multi_avg_n_probing_links_origin=np.average(multi_times_avg_n_probing_links_origin_array, axis=0)
mynetwork.logger_main.info("after average: avg diff from the real optimal path:")
mynetwork.logger_main.info(multi_avg_percentage_of_abs_diff_from_optimal)
mynetwork.logger_main.info("after average: std diff from the real optimal path:")
mynetwork.logger_main.info(multi_std_percentage_of_abs_diff_from_optimal)

#Monitoring Overhead - statistic for avg and std of the monitor overhead
multi_avg_n_probing_links_reduced=np.average(multi_times_avg_n_probing_links_reduced_array, axis=0)
multi_std_n_probing_links_reduced=np.std(multi_times_avg_n_probing_links_reduced_array, axis=0)
mynetwork.logger_main.info("after average: average probing links in reduced selected path:")
mynetwork.logger_main.info(multi_avg_n_probing_links_reduced)
mynetwork.logger_main.info("after average: std probing links in reduced selected path:")
mynetwork.logger_main.info(multi_std_n_probing_links_reduced)

#Freq. of optimal actions statistic of average and std of percentage_of_selected_optimal_path
multi_avg_percentage_of_select_optimal_path=np.average(multi_times_optimal_path_selected_percentage_array,axis=0)
multi_std_percentage_of_select_optimal_path=np.std(multi_times_optimal_path_selected_percentage_array,axis=0)
mynetwork.logger_main.info("after average: avg percentage of the optimal path selected:")
mynetwork.logger_main.info (multi_avg_percentage_of_select_optimal_path)
mynetwork.logger_main.info("after average: std percentage of the optimal path selected:")
mynetwork.logger_main.info (multi_std_percentage_of_select_optimal_path)

# rate of computed edge
multi_avg_percentage_of_compute_edge=np.average(multi_times_compute_edge_with_increasing_monitors,axis=0)
multi_std_percentage_of_compute_edge=np.std(multi_times_compute_edge_with_increasing_monitors,axis=0)
mynetwork.logger_main.info("after average: avg percentage of the computed edge")
mynetwork.logger_main.info (multi_avg_percentage_of_compute_edge)
mynetwork.logger_main.info("after average: std percentage of the computed edge:")
mynetwork.logger_main.info (multi_std_percentage_of_compute_edge)



#statistic for avg and std of path ocsillation stored in the file
multi_times_avg_path_oscilation_array, multi_times_std_path_oscilation_array=mynetwork.calculate_multi_times_serirs_results_avg_std(monitors_deployment_percentage,multi_times_path_ocilations_with_increasing_monitors, n, path_osc_dir,'path_osc')
mynetwork.plot_learning_error_of_total_edges_NSF(monitor_pert_list)
#mynetwork.plot_learning_error_of_total_opt_edges_NSF(monitor_pert_list)
#mynetwork.plot_path_oscillation_NSF(monitor_pert_list)


