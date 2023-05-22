import inline as inline
import matplotlib
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg
import sympy
from numpy.random import seed
from numpy.random import randint
from random import sample
import matplotlib.pyplot as plt

import math

class network_topology:

    def __init__(self, time, logger, directory):
        '''
        :param time: the total duration for MAB algorithm
        '''
        self.time=time
        self.Dict_edge_scales={}
        self.Dict_edge_delay_sample={}
        self.logger=logger
        self.directory=directory
    def graph_Generator(self,type, n, p):
        '''3
        :param type: the type of the topology
        :param n: the number of the nodes in the topology
        :param p: if it is 'ER' model, p:possibility of each edge in a complete graph to be present
                  if it is 'Barabasi' model, p: the #edges to connect the graph when adding a new node
        :return:  Graph G
        '''

        G=nx.Graph(name="my network")
        if type=="ER":  #generate a random ER topology model
            '''
            G=nx.erdos_renyi_graph(n, p)
            nx.write_gml(G,"topology/ER/er_graph_dot_file_%d_%s.gml" %(n, p))
            '''
            G=nx.read_graphml("topology/ER/er_graph_dot_file_%d_%s.gml" %(n, p))
        elif type=="Barabasi":
            '''
            #generate a new graph
            G = nx.barabasi_albert_graph(n,p)
            nx.write_gml(G, "topology/BR/br_graph_dot_file_%d_%s.gml" %(n, p))
            '''
            #read the graph from an existing file
            G=nx.read_gml("topology/BR/br_graph_dot_file_%d_2.gml" %(n))
        elif type=="Bics":
            G=nx.read_graphml('topology/Topology_Zoo/Bics.graphml')
            #G = nx.read_gml('topology/Topology_Zoo/Graph_Bics_10.gml')
        elif type=="BTN":
            G=nx.read_graphml('topology/Topology_Zoo/BeyondTheNetwork.graphml')
            degree_list = list(G.degree(list(G.nodes)))
            #compute the average node degree
            total_degree=0
            for edge_degree in degree_list:
                total_degree+=edge_degree[1]
            average_degree=total_degree/len(list(G.nodes))
            print("degree_list: %s" %(degree_list))
            print("total_degree: %d" %(total_degree))
            print("average degree is %f" %(average_degree))
            degree_one_nodes = []
            # it does not make sense to differenciate the end nodes from the internal nodes.
            # trim the node with degree 1
            for edge_degree in degree_list:
                if edge_degree[1] == 1:
                    degree_one_nodes.append(edge_degree[0])
            G.remove_nodes_from(degree_one_nodes)
        elif type=="Ntt":
            G = nx.read_gml('topology/Topology_Zoo/Ntt.gml')
            degree_list = list(G.degree(list(G.nodes)))
            degree_one_nodes = []
            # it does not make sense to differenciate the end nodes from the internal nodes.
            # trim the node with degree 1
            for edge_degree in degree_list:
                if edge_degree[1] == 1:
                    degree_one_nodes.append(edge_degree[0])
            G.remove_nodes_from(degree_one_nodes)
        elif type=="NSF":
            G = nx.read_gml('topology/Topology_Zoo/NSFnet.gml')
            degree_list = list(G.degree(list(G.nodes)))
            # compute the average node degree
            total_degree = 0
            for edge_degree in degree_list:
                total_degree += edge_degree[1]
            average_degree = total_degree / len(list(G.nodes))
            print("degree_list: %s" % (degree_list))
            print("total_degree: %d" % (total_degree))
            print("average degree is %f" % (average_degree))
            degree_one_nodes = []
            # it does not make sense to differenciate the end nodes from the internal nodes.
            # trim the node with degree 1
            for edge_degree in degree_list:
                if edge_degree[1] == 1:
                    degree_one_nodes.append(edge_degree[0])
            G.remove_nodes_from(degree_one_nodes)
        self.logger.info("Graph Created!")
        print(G.edges)
        for edge in G.edges:
            G[edge[0]][edge[1]]['weight']=1
        self.logger.info("all the edge weights in the graph are assigned to 1")
        self.construct_link_delay_distribution(G,type,n,p)
        #self.logger.info(f"Edge delay scales: {self.Dict_edge_scales}")
        self.draw_edge_delay_sample(G, type, n, p)
        self.assign_link_delay(G)
        #show the topology graph

        nx.draw(G, with_labels=True)
        plt.savefig(self.directory+"original_topology_%s_%s_%s.png" %(type,n,p), format="PNG")
        #graphy=plt.subplot(122)
        #nx.draw(G,pos=nx.circular_layout(G),node_color='r', edge_color='b')
        self.logger.info(G.name)
        self.logger.info("Graph Nodes: %s" %(G.nodes))
        self.logger.info("Graph Edges length: %d \n Graph Edges data: %s" %(len(G.edges),G.edges.data()))
        plt.close()
        return G


    def construct_link_delay_distribution(self,G,type,n,p):
        '''
        -The link delay distribution is defined as exponential model: f(x,λ)=λexp(-λx)
        1/λ is the mean of random variable x.the scale parameter in the exponential function is 1/λ. suppose every link delay
        is independent and all of them follow the exponential distribution with different parameter λ.
        Let's set the minimum link delay is 1 and the maximum link delay is 5, then we randomly select #G.edge numbers in [1,6) as
        the 1/λ vector(scale vector) of size #G.size, then we will construct #G.edge exponential distributions for each link.
        :param G: the generated graph
        :return: the scales vector for all the edge
        '''
        '''
        scales=np.random.randint(1, 10, len(G.edges))
        if type== "Barabasi" or type== "ER":
            np.savetxt('delay_exponential_samples/scales_%s_%s_%s.txt' % (type, n, p), scales)
        elif type=="Bics" or type=="BTN" or type=="Ntt":
            np.savetxt('delay_exponential_samples/scales_%s.txt' % (type), scales)
        '''
        scales=[]
        if type== "Barabasi" or type== "ER":
            y= np.loadtxt("delay_exponential_samples/scales_%s_%s_%s.txt" % (type, n, p))
            scales = np.array(y)
        elif type=="Bics" or type=="BTN" or type=="Ntt":
            y = np.loadtxt("delay_exponential_samples/scales_%s.txt" % (type))
            scales = np.array(y)
        elif type=="NSF":
            y = np.loadtxt("delay_exponential_samples/scales_%s_1500.txt" % (type))
            scales = np.array(y)
        self.logger.debug("Edge delay scales: %s" %(scales))
        i=0
        for edge in G.edges:
           self.Dict_edge_scales[edge]=scales[i]
           G[edge[0]][edge[1]]['delay_mean']=scales[i]
           i=i+1


    def draw_edge_delay_sample(self, G, type, n, p):
        '''
        This function draws a certain number of samples from the exponential distribution for each link delay
        :param G: the original graph G
        :param type: the graph type: {ER, Barabasi}
        :param n: the number of the nodes in the graph
        :param p: if it is ER, p is the possibility that a link in a fully connected graph exists in the current topology, here it is used to name the sample file
        :return: NULL, the samples will be saved in a file
        '''

        #read samples from an existing file

        if type== "Barabasi" or type=="ER":
            y = np.loadtxt("delay_exponential_samples/samples_%s_%s_%s.txt" %(type, n, p))
        elif type=="Bics" or type=="BTN" or type=="Ntt":
            y= np.loadtxt("delay_exponential_samples/samples_%s.txt" %(type))
        elif type=="NSF":
            y= np.loadtxt('delay_exponential_samples/samples_NSF_1500.txt')

        samples=np.array(y)
        for edge in G.edges:
            self.Dict_edge_delay_sample[edge]=[]
        i=0
        for edge in G.edges:
            self.Dict_edge_delay_sample[edge] =samples[i]
            i=i+1
        '''
        #generate the delay samples from the exponential distribution with the generated scales
        samples=[]
        for edge in G.edges:
            sample = np.random.exponential(scale=self.Dict_edge_scales[edge], size=(1, self.time+len(G.edges)))[0]
            self.Dict_edge_delay_sample[edge]=sample
            samples.append(sample)
        n_samples=np.array(samples)
        if type== "Barabasi" or type=="ER":
            np.savetxt('delay_exponential_samples/samples_%s_%s_%s.txt' %(type, n, p),n_samples)
        elif type=="Bics" or type=="BTN" or type=="Ntt" or type=="NSF":
            np.savetxt('delay_exponential_samples/samples_%s.txt' % (type), n_samples)
        '''
        self.logger.info("Draw %d delay examples from exponential distribution for each edge." %(self.time+len(G.edges)))

        average = [np.average(self.Dict_edge_delay_sample[edge]) for edge in G.edges]
        self.logger.info("edge delay sample average %s" %(average))

    def assign_link_delay(self,G):
        '''
        This function assigns the time series delay of each link every second
        :param G: Graph G
        :return: NULL
        '''
        #test
        '''
        for edge in G.edges:
             G[edge[0]][edge[1]]['delay']=G[edge[0]][edge[1]]['delay-mean']

        '''
        for edge in G.edges:
            #print(f"Dict_edge_delay_sample: {edge} {self.Dict_edge_delay_sample[edge]}")
            G[edge[0]][edge[1]]['delay'] = self.Dict_edge_delay_sample[edge][0]
            #print(f"type: {type(self.Dict_edge_delay_sample[edge])}")
            self.Dict_edge_delay_sample[edge]=np.delete(self.Dict_edge_delay_sample[edge],0)
        #self.logger.debug(f"{len(self.Dict_edge_delay_sample[edge])} samples left")
            #print(f"after deletion{self.Dict_edge_delay_sample[edge]}")
        #self.logger.debug(f"Assigned Delay {G.edges.data()}")

    def deploy_monitor(self,G, n, monitor_candidate_list):
        '''
        select the end nodes in which monitor can be deployed
        :param G: the graph (network topology)
        :param n: the number of the monitors will be deployed
        :param monitor_candidate_list: it can be in 3 cases:
                1. empty - the system will randomly choose n nodes in the graph to deploy the monitor
                2. sizeof(monitor_candidate_list)=n, user gives the nodes where to deploy the monitor
                3. sizeof(monitor_candidate_list)<n  user gives partial locations to deploy the monitor, the system will select
                                                     the rest (n-sizeof(monitor_candidate_list)
        :return: the nodes which are selected to deploy the monitor
        '''
        monitors = []
        self.logger.debug("n=%d monitor_candidate_list=%d" %(n, len(monitor_candidate_list)))
        if len(monitor_candidate_list) == n:
            monitors = monitor_candidate_list
        elif len(monitor_candidate_list) < n:
            monitors = monitor_candidate_list
            rest_nodes = [elem for elem in G.nodes if elem not in monitors]
            select = sample(rest_nodes, k=n - len(monitor_candidate_list))
            monitors = monitors + select
        self.logger.info("Monitors are deployed in nodes: %s" %(monitors))
        return monitors

    def getPath(self,G, monitors,weight):
        '''
        Get the path list with Dijkstra shortest path algorithm according to the specified weight factor
        :param G:
        :param monitors:
        :return:
        '''
        nodepairs = [(monitors[i], monitors[j]) for i in range(len(monitors)) for j in range(i + 1, len(monitors))]
        path_list = []
        try:
            for n1, n2 in nodepairs:
                shortest_path = nx.shortest_path(G, source=n1, target=n2, weight=weight, method='dijkstra')
                pathpair = []
                [pathpair.append((shortest_path[i], shortest_path[i + 1])) for i in range(len(shortest_path) - 1)]
                path_list.append(pathpair)
                ''' #compute all the possible paths and selected the first one
                paths=nx.all_simple_paths(G,n1,n2)
                for path in map(nx.utils.pairwise,paths):
                    #print(f"path from {n1} to {n2}: {list(path)}")
                    path_list.append(list(path))
                    break
                '''
            # print(f"end to end paths:{path_list}")
            return path_list
        except Exception as e:
            self.logger.error(str(e) + "occured, the graph is disconnected, please regenerate the graph")

    def trimNetwrok(self, G, monitors):
        '''
        this function aims to eliminate all the links that are not covered in any simple path between any pair of monitors
        :param G: the original topology
        :return: the trimed topology
        '''
        self.logger.info("Trim the network...............")
        all_paths_set=set()
        nodepairs = [(monitors[i], monitors[j]) for i in range(len(monitors)) for j in range(i + 1, len(monitors))]
        # print(f"end to end nodepairs: {nodepairs}")
        path_count_among_every_monitor_pair=[]
        for n1, n2 in nodepairs:
            #compute all the possible paths and selected the first one
            paths=nx.all_simple_paths(G,n1,n2)
            path_count_among_every_monitor_pair.append(len(list(paths)))
            for path in map(nx.utils.pairwise, paths):
               for edge in list(path):
                    all_paths_set.add(edge)
            #self.logger.info(f"there are {len(all_paths_set)} between monitor pair node {n1} {n2}: they are:")
            #self.logger.info(paths)
        #print(f"end to end paths set: {all_paths_set}")
        average_path_count=np.average(np.array(path_count_among_every_monitor_pair))
        self.logger.info("average_path_count among monitors: %f" %(average_path_count))
        edges=list(G.edges)
        uncovered_edges=[]
        for edge_e in edges:
                edge_r=(edge_e[1],edge_e[0])
                if edge_e not in all_paths_set and edge_r not in all_paths_set:
                    #print(f"1 {edge_e}")
                    uncovered_edges.append(edge_e)

        self.logger.info("uncovered edges: %s" %(uncovered_edges))
        trimedG=G.copy()   #store the original network topology
        for edge in uncovered_edges:
            trimedG.remove_edge(*edge)

        #G.remove_nodes_from(['8','10'])

        plt.figure()
        #nx.draw(trimedG, with_labels=True)
        plt.savefig(self.directory + "trimed_topology", format="PNG")
        self.logger.info("after elimination the Graph has %d edges  %s" %(len(list(trimedG.edges)),list(trimedG.edges)))
        return trimedG