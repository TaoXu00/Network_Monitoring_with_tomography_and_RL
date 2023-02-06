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
        for m_p in [10,20,30,40,50]:
        #for m_p in [20, 30]:
            monitors_deployment_percentage.append(m_p)
            n = int((m_p / 100) * len(G.nodes))
            #if m_p==10:
            #    n=3
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
        return optimal_path_selected_percentage_list, avg_diff_of_delay_from_optimal_list, total_edge_mse_list_with_increasing_monitors, total_optimal_edges_mse_list_with_increasing_monitors, monitors_deployment_percentage, average_probing_links_origin_list, average_probing_links_reduced_list, rate_of_optimal_actions_list_with_increasing_monitors, path_oscilation_list_with_increasing_monitors, traffic_overhead_every_200_iterations_with_increasing_monitors

    def plot_edge_computed_rate_bar_with_different_topology_size(self):
        self.plotter.plot_edge_computed_rate_with_different_topology_size()

    def plot_final_result(self, mynetwork):
        #plot the scalability performance in Barabasi 50 with 30% monitors deployed but varies the network size
        monitors_deployment_percentage = [10,20,30,40,50]
        #myapproach_optimal_path_selected_rate = [0.7303425, 0.784525, 0.820643333, 0.838918553, 0.902852585]
        #myapproach_optimal_path_selected_rate=[0.6682375, 0.708409445 ,0.721732615, 0.751597365, 0.80618583]

        subito_op_rate = [0.709256667, 0.75291074, 0.787211903, 0.80995684, 0.869051943]
        UCB1_op_rate = [0.56466, 0.557230555, 0.56462738, 0.56482263, 0.57180575]
        subito_perfect_op_rate = [0.8338175, 0.864745, 0.86949619, 0.87432776, 0.8759815]
        BoundNT_op_rate=[0.48122,  0.42990222, 0.41871429, 0.39760421, 0.375116]

        UCB1_diff = [4.55981845, 4.32474493, 4.284358305, 4.25667079, 4.15141293]
        subito_diff = [3.572746567, 2.762555307, 2.438297613, 2.284643933, 1.743578883]
        subito_perfect_diff = [2.236884885, 1.71805823, 1.60399359, 1.53417783, 1.49493546]
        BoundNT_diff=[5.38756999, 5.86999576, 6.06494927, 6.2990515, 6.444486]

        UCB1_traffic_overhead=[45.6643095,197.969513,459.798732,834.551768,1305.78879]
        #subito_MAB_trffic_overhead=[ 36.44358334,166.6565, 384.0626667,	700.6428667, 1107.435267]
        subito_NT_traffic_overhead=[31.25533334, 103.3804,	146.1748,	175.6659333, 190.4278667]
        bound_NT_traffic_overhead=[30.3286667,79.7326667,129.9286,178.825867,220.262267]

        #mynetwork.plotter.plot_percentage_of_optimal_path_selected_rate_BR_50nodes(monitors_deployment_percentage, subito_op_rate, UCB1_op_rate, subito_perfect_op_rate)
        #mynetwork.plotter.plot_abs_delay_of_optimal_path_selected_from_mean_BR_50nodes(monitors_deployment_percentage, subito_diff, UCB1_diff, subito_perfect_diff)
        mynetwork.plotter.plot_percentage_of_optimal_path_selected_rate_BR_50nodes_line(monitors_deployment_percentage, subito_op_rate, UCB1_op_rate,subito_perfect_op_rate,BoundNT_op_rate)
        mynetwork.plotter.plot_abs_delay_of_optimal_path_selected_from_mean_BR_50nodes_line(monitors_deployment_percentage, subito_diff, UCB1_diff, subito_perfect_diff, BoundNT_diff)

        mynetwork.plotter.plot_traffic_overhead_BR_50nodes(monitors_deployment_percentage, bound_NT_traffic_overhead,subito_NT_traffic_overhead, UCB1_traffic_overhead)
        #plot the scalability performance of network size 20, 40, 60, 80 nodes with fixed 30% monitors deployed
        topology_size=[20, 40, 60, 80]
        subito_op_rate = [0.86708, 0.78523, 0.729153167, 0.666417933]
        UCB1_op_rate = [0.50193667, 0.44657879, 0.34701209, 0.3847663]
        subito_perfect_op_rate = [0.88412667, 0.85507955, 0.77260229, 0.74318261]
        boundNT_op_rate=[0.47973333, 0.4293, 0.37392418, 0.34462681]

        UCB1_diff = [4.30608774, 6.11897566, 6.15564568, 8.3113577]
        subito_diff = [1.45996846, 2.430924276, 2.498190838, 3.326673313]
        subito_perfect_diff = [1.29733573, 1.9665792, 2.29629007, 2.98189548]
        boundNT_diff=[4.73952412, 5.84895738, 6.52524591, 7.91529036]

        subito_MAB_trffic_overhead=[48.646, 255.37633333, 527.29533333, 1093.01733333]
        subito_NT_traffic_overhead=[26.62933333, 107.21466667, 138.905, 258.91733333]
        UCB1_traffic_overhead=[46.46057372, 271.77565043, 665.71047365, 1454.89766511]

        #mynetwork.plotter.plot_percentage_of_optimal_path_selected_rate_for_various_network_size(topology_size, subito_op_rate, UCB1_op_rate, subito_perfect_op_rate)
        #mynetwork.plotter.plot_abs_delay_of_optimal_path_selected_for_various_network_size(topology_size, subito_diff, UCB1_diff, subito_perfect_diff)
        mynetwork.plotter.plot_percentage_of_optimal_path_selected_rate_for_various_network_size_line(topology_size,subito_op_rate,UCB1_op_rate,subito_perfect_op_rate,boundNT_op_rate)
        mynetwork.plotter.plot_abs_delay_of_optimal_path_selected_for_various_network_size_line(topology_size, subito_diff, UCB1_diff,subito_perfect_diff, boundNT_diff)
        mynetwork.plotter.plot_traffic_overhead_for_various_network_size(topology_size, subito_MAB_trffic_overhead, subito_NT_traffic_overhead, UCB1_traffic_overhead)


        #plot the experiments for real infrastructure
        monitors_deployment_percentage = [10, 20, 30, 40, 50]
        myapproach_optimal_path_selected_rate = [0.8730325,	0.865852223, 0.84571643, 0.835979885,0.843303463]
        UCB1_op_rate = [0.67076,	0.68020833,	0.67219667,	0.66329536,	0.67381338]
        subito_perfect_op_rate=[0.876075,	0.9192025,	0.911354763,	0.908002727,	0.91525641]
        subito_op_rate=[0.84235, 0.874365,	0.859673215, 0.88029, 0.887951095]
        UCB1_diff = [3.92730099,	3.55005621,	3.76985005,	4.01648821,	4.05722965]
        subito_perfect_diff= [1.704276463, 1.125456645, 1.099625047, 1.085538627, 1.023573517]
        subito_diff=[2.266471527, 1.8255659,	1.984103853, 1.734049937, 1.540486353]

        subito_MAB_trffic_overhead = [14.578,  41.62033333, 118.732,  247.68466667, 408.48166667]
        subito_NT_traffic_overhead = [14.578,  28.36566667, 50.804, 64.11766667, 69.70233333]
        UCB1_traffic_overhead=[] #to run

        mynetwork.plotter.plot_percentage_of_optimal_path_selected_rate_BTN(monitors_deployment_percentage, subito_op_rate, UCB1_op_rate, subito_perfect_op_rate )
        mynetwork.plotter.plot_abs_delay_of_optimal_path_selected_from_mean_BTN(monitors_deployment_percentage,subito_diff,UCB1_diff, subito_perfect_diff)
    def plot_traffic_overhead_of_subito(self):
        multi_times_avg_traffic_overhead_every_200_iterations_with_increasing_monitors=[]
        multi_times_avg_traffic_overhead_every_200_iterations_with_increasing_monitors.append([23.88, 23.7225, 23.725, 23.6975, 23.799, 23.781666666666666, 23.826428571428572, 23.844375, 23.850555555555555, 23.8665, 23.89727272727273, 23.855, 23.860384615384614, 23.84464285714286, 23.845666666666666])
        multi_times_avg_traffic_overhead_every_200_iterations_with_increasing_monitors.append([85.345, 86.1, 86.45666666666666, 86.28125, 85.929, 85.51833333333333, 85.405, 85.40875, 85.46888888888888, 85.375, 85.31090909090909, 85.35958333333333, 85.425, 85.49035714285715, 85.40033333333334])
        multi_times_avg_traffic_overhead_every_200_iterations_with_increasing_monitors.append([143.095, 141.91, 141.45833333333334, 141.43375, 141.331, 141.455, 141.49642857142857, 141.3525, 141.38222222222223, 141.269, 141.20272727272726, 141.28625, 141.28192307692308, 141.185, 141.30733333333333])
        multi_times_avg_traffic_overhead_every_200_iterations_with_increasing_monitors.append([199.185, 196.55, 196.16333333333333, 195.39875, 195.389, 195.3325, 195.67785714285714, 195.510625, 195.2588888888889, 195.1845, 195.3318181818182, 195.22666666666666, 195.1380769230769, 194.9882142857143, 194.85866666666666])
        multi_times_avg_traffic_overhead_every_200_iterations_with_increasing_monitors.append([222.245, 223.7175, 223.51, 223.69875, 223.841, 223.6375, 223.72, 223.84875, 223.62722222222223, 223.5515, 223.49318181818182, 223.49041666666668, 223.44307692307692, 223.33714285714285, 223.45566666666667])
        self.plotter.plot_avg_traffic_overhead_every_200_iterations([10,20,30,40,50], multi_times_avg_traffic_overhead_every_200_iterations_with_increasing_monitors)
    def plot_path_oscillation_boundNT(self):
        multi_times_avg_path_oscillation_every_200_iterations_with_increasing_monitors = []
        multi_times_avg_path_oscillation_every_200_iterations_with_increasing_monitors.append([4.3, 5.3, 5.1, 4.7, 5.2, 4.6, 5.6, 5.4, 4.5, 4.9, 5.8, 4.6, 5.0, 5.1, 4.1])
        multi_times_avg_path_oscillation_every_200_iterations_with_increasing_monitors.append([13.622222222222222, 13.622222222222222, 13.355555555555556, 11.622222222222222, 13.555555555555555, 12.28888888888889, 13.533333333333333, 12.28888888888889, 12.28888888888889, 13.28888888888889, 13.28888888888889, 12.8, 13.733333333333333, 12.666666666666666, 12.666666666666666])
        multi_times_avg_path_oscillation_every_200_iterations_with_increasing_monitors.append([21.571428571428573, 20.714285714285715, 20.933333333333334, 22.00952380952381, 20.752380952380953, 21.62857142857143, 20.0, 21.523809523809526, 20.99047619047619, 21.542857142857144, 20.533333333333335, 21.20952380952381, 22.6, 21.238095238095237, 20.304761904761904])
        multi_times_avg_path_oscillation_every_200_iterations_with_increasing_monitors.append([27.18421052631579, 27.03684210526316, 26.873684210526317, 27.731578947368423, 28.33157894736842, 26.978947368421053, 27.489473684210527, 28.83684210526316, 27.594736842105263, 26.126315789473683, 27.873684210526317, 27.46315789473684, 28.442105263157895, 27.268421052631577, 27.03684210526316])
        multi_times_avg_path_oscillation_every_200_iterations_with_increasing_monitors.append([32.166666666666664, 31.706666666666667, 30.45, 32.06666666666667, 33.32, 32.32666666666667, 31.94333333333333, 31.756666666666668, 31.25, 31.356666666666666, 31.803333333333335, 30.946666666666665, 30.923333333333332, 31.376666666666665, 32.04])
        np.savetxt("BoundNT_path_oscillation_BR50_10%-50%.txt", multi_times_avg_path_oscillation_every_200_iterations_with_increasing_monitors)
        self.plotter.plot_avg_path_oscilation_every_200_times([10, 20, 30, 40, 50],multi_times_avg_path_oscillation_every_200_iterations_with_increasing_monitors)

    def plot_path_oscillation(self):
        monitor_pert=[10,20,30,40,50]
        BoundNT=np.loadtxt("path_oscillation/BoundNT_path_oscillation_BR50_10%-50%.txt")
        Subito=np.loadtxt("path_oscillation/Subito_ocsillition_every_200_times_BR50_10%-50%.txt")
        UCB1=np.loadtxt("path_oscillation/UBC1_ocsillition_every_200_times_BR50_10%-50%_baseline.txt")
        self.plotter.plot_avg_path_oscilation_every_200_times_withname(monitor_pert,Subito,"Subito_ocsillition_every_200_times_BR50_10%-50%")
        self.plotter.plot_avg_path_oscilation_every_200_times_withname(monitor_pert,BoundNT, "BoundNT_path_oscillation_BR50_10%-50%")
        self.plotter.plot_avg_path_oscilation_every_200_times_withname(monitor_pert,UCB1,"UBC1_ocsillition_every_200_times_BR50_10%-50%_baseline")

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


mynetwork=main(3000)
mynetwork.plot_final_result(mynetwork)
#mynetwork.plotter.plot_total_edge_delay_mse_with_increasing_monitor_training_from_file([10,20,30,40,50],"mse_results/links_delay_during_training_with_different_monitor_size_total.txt")
#mynetwork.plot_path_oscillation_boundNT()
#mynetwork.plot_path_oscillation()
'''
while(i<n):
    mynetwork=main(3000)
    G =mynetwork.creat_topology(topo_type, num_node, degree)
    #mynetwork.tomography_verification(G,'weight')   #here the assigned delay should be 1, place modify the topo.assign_link_delay() function
    optimal_path_selected_percentage_list, avg_diff_of_delay_from_optimal_list,total_edge_mse_list_with_increasing_monitors, total_optimal_edges_mse_list_with_increasing_monitors,monitors_deployment_percentage, average_probing_links_origin_list, average_probing_links_reduced_list, rate_of_optimal_actions_list_with_increasing_monitors, path_oscilation_list_with_increasing_monitors, traffic_overhead_every_200_iterations_with_increasing_monitors = mynetwork.MAB_with_increasing_monitors(G,topo_type,len(G.nodes),degree, llc_factor)
    #print("n=%d" %(i))
    #print(optimal_path_selected_percentage_list,avg_diff_of_delay_from_optimal_list)
    if i==0:
        multi_times_mse_total_link_delay_array=np.array(total_edge_mse_list_with_increasing_monitors, dtype=float)
        multi_times_mse_total_optimal_link_delay_array=np.array(total_optimal_edges_mse_list_with_increasing_monitors, dtype=float)
        multi_times_optimal_path_selected_percentage_array=np.array([optimal_path_selected_percentage_list])
        multi_times_avg_diff_of_delay_from_optimal_array=np.array([avg_diff_of_delay_from_optimal_list])
        multi_times_avg_n_probing_links_origin_array=np.array([average_probing_links_origin_list])
        multi_times_avg_n_probing_links_reduced_array=np.array([average_probing_links_reduced_list])
        multi_times_rate_of_optimal_actions_list_with_increasing_monitors=np.array(rate_of_optimal_actions_list_with_increasing_monitors)
        multi_times_path_ocilations_with_increasing_monitors=np.array(path_oscilation_list_with_increasing_monitors)
        multi_times_traffic_overhead_every_200_iterations_with_increasing_monitors = np.array(traffic_overhead_every_200_iterations_with_increasing_monitors)
    else:
        current_mse_arrary=np.array(total_edge_mse_list_with_increasing_monitors)
        current_mse_optimal_edges_arrary=np.array(total_optimal_edges_mse_list_with_increasing_monitors)
        current_rate_of_optimal_actions_list_with_increasing_monitor_size=np.array(rate_of_optimal_actions_list_with_increasing_monitors)
        multi_times_mse_total_link_delay_array=np.add(multi_times_mse_total_link_delay_array,current_mse_arrary)
        multi_times_mse_total_optimal_link_delay_array=np.add(multi_times_mse_total_optimal_link_delay_array,current_mse_optimal_edges_arrary)
        multi_times_optimal_path_selected_percentage_array=np.append(multi_times_optimal_path_selected_percentage_array,np.array([optimal_path_selected_percentage_list]),axis=0)
        multi_times_avg_diff_of_delay_from_optimal_array=np.append(multi_times_avg_diff_of_delay_from_optimal_array,np.array([avg_diff_of_delay_from_optimal_list]), axis=0)
        multi_times_avg_n_probing_links_origin_array=np.append(multi_times_avg_n_probing_links_origin_array,np.array([average_probing_links_origin_list]))
        multi_times_avg_n_probing_links_reduced_array=np.append(multi_times_avg_n_probing_links_reduced_array, np.array([average_probing_links_reduced_list]))
        multi_times_path_ocilations_with_increasing_monitors = np.add(multi_times_path_ocilations_with_increasing_monitors,np.array(path_oscilation_list_with_increasing_monitors))
        multi_times_rate_of_optimal_actions_list_with_increasing_monitors=np.add(multi_times_rate_of_optimal_actions_list_with_increasing_monitors, np.array(current_rate_of_optimal_actions_list_with_increasing_monitor_size))
        multi_times_traffic_overhead_every_200_iterations_with_increasing_monitors = np.add(multi_times_traffic_overhead_every_200_iterations_with_increasing_monitors, np.array(traffic_overhead_every_200_iterations_with_increasing_monitors))
    i += 1


multi_times_avg_mse_total_link_delay_array=multi_times_mse_total_link_delay_array/n
multi_times_avg_mse_total_optimal_links_delay_array=multi_times_mse_total_optimal_link_delay_array/n
multi_times_avg_path_oscilation_array=multi_times_path_ocilations_with_increasing_monitors/n
multi_times_avg_traffic_overhead_every_200_iterations_with_increasing_monitors=multi_times_traffic_overhead_every_200_iterations_with_increasing_monitors/n
np.savetxt(mynetwork.directory + 'links_delay_during_training_with_different_monitor_size_total.txt', multi_times_avg_mse_total_link_delay_array)
np.savetxt(mynetwork.directory + 'optimal_links_delay_during_training_with_different_monitor_size_total.txt', multi_times_avg_mse_total_optimal_links_delay_array)
mynetwork.logger_main.info("Statistics:")
mynetwork.logger_main.info("Before average: percentage of the optimal path selected:")
mynetwork.logger_main.info(multi_times_optimal_path_selected_percentage_array)
mynetwork.logger_main.info("Before average: diff from the real optimal path: ")
mynetwork.logger_main.info(multi_times_avg_diff_of_delay_from_optimal_array)
mynetwork.logger_main.info("Before average: average probing links in original selected path:")
mynetwork.logger_main.info(multi_times_avg_n_probing_links_origin_array)
mynetwork.logger_main.info("Before average: average probing links in reduced selected path:")
mynetwork.logger_main.info(multi_times_avg_n_probing_links_reduced_array)
mynetwork.logger_main.info("Before average: rate of the optimal actions shape: ")
mynetwork.logger_main.info(multi_times_rate_of_optimal_actions_list_with_increasing_monitors.shape)


multi_avg_percentage_of_select_optimal_path=np.average(multi_times_optimal_path_selected_percentage_array,axis=0)
multi_avg_percentage_of_abs_diff_from_optimal=np.average(multi_times_avg_diff_of_delay_from_optimal_array,axis=0)
multi_avg_n_probing_links_origin=np.average(multi_times_avg_n_probing_links_origin_array, axis=0)
multi_avg_n_probing_links_reduced=np.average(multi_times_avg_n_probing_links_reduced_array, axis=0)
multi_avg_optimal_actions_with_increasing_monitors=multi_times_rate_of_optimal_actions_list_with_increasing_monitors/n
mynetwork.logger_main.info("after average: percentage of the optimal path selected:")
mynetwork.logger_main.info (multi_avg_percentage_of_select_optimal_path)
mynetwork.logger_main.info("after average: diff from the real optimal path:")
mynetwork.logger_main.info(multi_avg_percentage_of_abs_diff_from_optimal)
mynetwork.logger_main.info("after average: average probing links in original selected path:")
mynetwork.logger_main.info(multi_avg_n_probing_links_origin)
mynetwork.logger_main.info("after average: average probing links in reduced selected path:")
mynetwork.logger_main.info(multi_avg_n_probing_links_reduced)
mynetwork.logger_main.info("after average: rate of optimal actions shape:")
mynetwork.logger_main.info(multi_avg_optimal_actions_with_increasing_monitors.shape)
mynetwork.logger_main.info("after average: path oscilations")
mynetwork.logger_main.info(multi_times_avg_path_oscilation_array)

mynetwork.plotter.plot_total_edge_delay_mse_with_increasing_monitor_training(monitors_deployment_percentage,multi_times_avg_mse_total_link_delay_array)
mynetwork.plotter.plot_total_optimal_edge_delay_mse_with_increasing_monitor_training(monitors_deployment_percentage,multi_times_avg_mse_total_optimal_links_delay_array)
# self.plotter.plot_edge_exporation_times_with_differrent_monitor_size(G,total_edge_exploration_during_training_list)
mynetwork.plotter.plot_optimal_path_selected_percentage_list_with_increasing_monitors(monitors_deployment_percentage, multi_avg_percentage_of_select_optimal_path)
mynetwork.plotter.plot_abs_diff_path_delay_from_the_optimal(monitors_deployment_percentage,multi_avg_percentage_of_abs_diff_from_optimal )
mynetwork.plotter.plot_avg_path_oscilation_every_200_times(monitors_deployment_percentage,multi_times_avg_path_oscilation_array)
np.savetxt("Subito_ocsillition_every_200_times_BR50_10%-50%.txt",multi_times_avg_path_oscilation_array)
mynetwork.plotter.plot_avg_optimal_actions_every_200_times(monitors_deployment_percentage,multi_avg_optimal_actions_with_increasing_monitors)
mynetwork.plotter.plot_avg_traffic_overhead_every_200_iterations(monitors_deployment_percentage, multi_times_avg_traffic_overhead_every_200_iterations_with_increasing_monitors)
'''
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


