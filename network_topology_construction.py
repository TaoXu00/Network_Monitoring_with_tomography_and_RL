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
    def graph_Generator(self, n, p):
        '''
        :param n: the number of the nodes in the topology
        :param p: possibility of each edge in a complete graph to be present
        :return G:
        '''
        G=nx.Graph(name="my network")
        #G.add_nodes_from([A,'B','C','D'])
        '''Topology 1
        G.add_edges_from([('A','D',{'delay':1}),('D','B',{'delay':0.5} ), ('A','C', {'delay':2})])
        endnodes=['A','B','C']
        '''
        ''' Topology 2
        G.add_edges_from([('A', 'C', {'delay': 0.5}), ('B', 'C', {'delay': 1}), ('C', 'D', {'delay': 2}),
                          ('D', 'E', {'delay': 0.5}),('D', 'F', {'delay': 1})])
        endnodes = ['A', 'B', 'E','F']
        '''
        '''Topology 3
        G.add_edges_from([('A', 'C', {'delay': 0.5}), ('B', 'C', {'delay': 1}), ('C', 'D', {'delay': 2}),
                          ('D', 'E', {'delay': 0.5}), ('D', 'F', {'delay': 1}),('C', 'G', {'delay': 0.1}),
                          ('G', 'D', {'delay': 0.1}), ('C', 'H', {'delay': 0.1}),('D', 'H', {'delay': 0.1}),
                          ('E', 'I', {'delay': 0.5}),])
        endnodes = ['A', 'B', 'I', 'F']
        '''

        #Topology 4: randomized network topology
        #G=nx.erdos_renyi_graph(n, p)
        #nx.write_gml(G,"graph_dot_file_50.gml")

        #G = nx.read_gml("graph_dot_file.gml")
        G = nx.read_gml("graph_dot_file_50.gml")
        self.logger.info("Graph Created!")
        #seed(1)

        #Topology 5, fixed graph for multi-armed-bandits algorithm

        #G.add_weighted_edges_from([(0, 1, 1), (0, 2, 1), (0, 3, 1), (1, 4, 1),(1,2,1),(2,5,1), (2, 3, 1),(3,6,1), (4,7,1),(4,5,1),(5,6,1),(5,8,1),(6,9,1),(7,10,1),
        #                           (7,8,1), (8,10,1),(8,9,1), (9,10,1)])
        for edge in G.edges:
            #G[edge[0]][edge[1]]['delay']=randint(1,10)
            G[edge[0]][edge[1]]['weight']=1
        self.logger.info("all the edge weights in the graph are assigned to 1")
        self.construct_link_delay_distribution(G)
        self.logger.info(f"Edge delay scales: {self.Dict_edge_scales}")
        self.draw_edge_delay_sample(G)
        self.assign_link_delay(G)
        #show the topology graph
        nx.draw(G, with_labels=True)
        plt.savefig(self.directory+"original_topology", format="PNG")
        #graphy=plt.subplot(122)
        #nx.draw(G,pos=nx.circular_layout(G),node_color='r', edge_color='b')

        self.logger.info(G.name)
        self.logger.info(f"Graph Nodes: {G.nodes}")
        self.logger.info(f"Graph Edges length:{len(G.edges)}\nGraph Edges data: {G.edges.data()}")
        #print(nx.to_numpy_matrix(G,nodelist=['A','B','C','D'])) #get the adjacent matrix of the graph
        return G


    def construct_link_delay_distribution(self,G):
        '''
        -The link delay distribution is defined as exponential model: f(x,λ)=λexp(-λx)
        1/λ is the mean of random variable x.the scale parameter in the exponential function is 1/λ. suppose every link delay
        is independent and all of them follow the exponential distribution with different parameter λ.
        Let's set the minimum link delay is 1 and the maximum link delay is 5, then we randomly select #G.edge numbers in [1,6) as
        the 1/λ vector(scale vector) of size #G.size, then we will construct #G.edge exponential distributions for each link.
        :param G: the generated graph
        :return: the scales vector for all the edge
        '''

        #scales=np.random.randint(1, 10, len(G.edges))
        #print(f"scales: {scales}")
        #scales=np.array([3, 6, 7, 2, 9, 8, 8, 9, 8, 3, 2, 8, 5, 3, 9, 9, 3, 2, 8, 5, 4, 6, 3, 4, 1, 4, 7, 7, 5, 7, 7, 6, 6, 3, 5, 4, 3,
 #9, 3, 7, 3, 6, 2, 8])
        #scales=np.array([15, 35, 27, 74,  1, 27, 61, 29, 40, 76, 19, 29, 36, 78, 20, 25, 85, 42, 38, 83, 80, 62, 41, 57,

        #48, 87, 77, 43, 29, 83, 25, 76, 49, 65, 51, 61, 92, 85, 79, 37, 60,  7, 99, 47])

        scales = np.array([8, 3, 3, 6, 4, 1, 7, 1, 7, 4, 8, 5, 2, 1, 9, 6, 4, 4, 1, 2, 6, 4, 1, 3, 9, 9, 7, 8, 9, 3, 7, 4, 6, 4, 9, 1, 8,
             5, 6, 1, 7, 9, 1, 6, 6, 2, 8, 4, 1, 7, 8, 4, 5, 6, 7, 6, 6, 7, 7, 5, 6, 8, 7, 7, 3, 5, 6, 8, 7, 5, 5, 9, 1, 4,
             1, 4, 1, 9, 7, 3, 3, 7, 8, 9, 7, 1, 3, 6, 5, 8, 5, 4, 2, 6, 4, 6, 2, 1, 4, 3, 9, 2, 9, 8, 6, 6, 9, 6, 5, 2, 3,
             4, 2, 7, 5, 1, 8, 8, 1, 3, 1, 3, 5, 4, 2, 5, 2, 7, 2, 6, 8, 3, 2, 4, 7, 4, 3, 9, 5, 6, 2, 3, 8, 1, 8, 5, 2, 3,
             9, 6, 3, 6, 2, 3, 4, 8, 7, 6, 1, 7, 3, 1, 9, 2, 8, 2, 4, 2, 9, 6, 1, 5, 2, 4, 3, 1, 7, 2, 5, 5, 7, 4, 9, 8, 2,
             1, 2, 2, 4, 2, 8, 3, 6, 4, 4, 9, 8,2, 6, 2, 1, 9, 6, 1, 7, 4, 2, 6, 3, 8, 8, 2, 8, 6, 9, 3, 5, 9, 9, 8, 6, 6,
             3, 5, 9, 6, 9, 4, 3, 3, 7, 3, 3, 7, 5, 5, 6, 6, 2, 5, 6, 4, 9, 3, 7, 1, 5, 4, 3, 5, 8, 7, 3, 9, 3,1, 3, 7, 8,
             6, 7, 6, 6, 2, 6, 5, 3, 7, 9, 8, 4, 7, 4, 1, 8, 7, 3, 3, 4, 5, 1, 1, 1, 1, 8, 6, 3, 9, 9, 4, 6, 4, 3, 2, 7, 7,
             9, 1, 6, 8, 9, 1, 7, 8, 9, 2, 5, 1, 4, 6, 7, 7, 7, 7, 6, 1, 8, 3, 3, 9, 6, 5, 9, 6, 7, 2])

        i=0
        for edge in G.edges:
           self.Dict_edge_scales[edge]=scales[i]
           G[edge[0]][edge[1]]['delay_mean']=scales[i]
           i=i+1
        #print(f"Dict_edge_scales:{self.Dict_edge_scales}")

    def draw_edge_delay_sample(self, G):
        for edge in G.edges:
            self.Dict_edge_delay_sample[edge]=[]
        for edge in G.edges:
            # G[edge[0]][edge[1]]['delay'] = np.random.exponential(scale=Dict_edge_scales[edge], size=1)[0]
            scale=self.Dict_edge_scales[edge]
            sample = np.random.exponential(scale=self.Dict_edge_scales[edge], size=(1, self.time))[0]
            self.Dict_edge_delay_sample[edge]=sample
            #print(f"draw sample for edge:({edge[0]},{edge[1]})")
            #print(self.Dict_edge_delay_sample [edge])
            #print(np.average(sample))
        self.logger.info(f"Draw {self.time} delay examples from exponential distribution for each edge.")
        average = [np.average(self.Dict_edge_delay_sample[edge]) for edge in G.edges]
        self.logger.info(f"edge delay sample average {average}")


    def assign_link_delay(self,G):
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
            #print(f"after deletion{self.Dict_edge_delay_sample[edge]}")
        self.logger.debug(f"Assigned Delay {G.edges.data()}")


    def deploy_monitor(self,G, n, monitor_candidate_list):
        '''
        select on which endnodes it will deploy the monitor
        :param G: the graph (network topology)
        :param n: the number of the monitors will be deployed
        :param monitor_candidate_list: it can be in 3 cases:
                1. empty - the system will randomly choose n nodes in the graph to deploy the monitor
                2. sizeof(monitor_candidate_list)=n, user gives the nodes where to deploy the monitor
                3. sizeof(monitor_candidate_list)<n  user gives partial locations to deploy the monitor, the system will select
                                                     the rest (n-sizeof(monitor_candidate_list)
        :return: the nodes which are selected to deploy the monitor
        '''
        ''' monitors for topology 1
        monitors=G.nodes
        '''
        # monitors for topology 2
        # monitors=['A','B','E','F']  #all the end nodes are selected to deploy the monitor

        # monitors for topology 3
        # monitors=['A','B','I','F']
        #print(G.nodes)
        monitors = []
        self.logger.debug(f"n={n} monitor_candidate_list={len(monitor_candidate_list)}")
        if len(monitor_candidate_list) == n:
            monitors = monitor_candidate_list
        elif len(monitor_candidate_list) < n:
            monitors = monitor_candidate_list
            rest_nodes = [elem for elem in G.nodes if elem not in monitors]
            select = sample(rest_nodes, k=n - len(monitor_candidate_list))
            monitors = monitors + select
        self.logger.info(f"Monitors are deployed in nodes: {monitors}")

        return monitors

    def getPath(self,G, monitors):
        nodepairs = [(monitors[i], monitors[j]) for i in range(len(monitors)) for j in range(i + 1, len(monitors))]
        # print(f"end to end nodepairs: {nodepairs}")
        path_list = []
        try:
            for n1, n2 in nodepairs:
                shortest_path = nx.shortest_path(G, source=n1, target=n2, weight='delay', method='dijkstra')
                # print(f"shortest path: {shortest_path}")
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
        all_paths_list=[]
        all_paths_set=set()
        nodepairs = [(monitors[i], monitors[j]) for i in range(len(monitors)) for j in range(i + 1, len(monitors))]
        # print(f"end to end nodepairs: {nodepairs}")
        for n1, n2 in nodepairs:
            #compute all the possible paths and selected the first one
            paths=nx.all_simple_paths(G,n1,n2)
            #print(f"totoal path number: {len( list(paths))}")
            for path in map(nx.utils.pairwise, paths):
               for edge in list(path):
                    all_paths_set.add(edge)
            self.logger.info(f"there are {len(all_paths_set)} between monitor pair node {n1} {n2}: they are:")
            self.logger.info(paths)
        #print(f"end to end paths set: {all_paths_set}")
        edges=list(G.edges)
        uncovered_edges=[]
        for edge_e in edges:
                edge_r=(edge_e[1],edge_e[0])
                if edge_e not in all_paths_set and edge_r not in all_paths_set:
                    #print(f"1 {edge_e}")
                    uncovered_edges.append(edge_e)

        self.logger.info(f"uncovered edges: {uncovered_edges}")
        old_G=G.copy()   #store the original network topology
        for edge in uncovered_edges:
            G.remove_edge(*edge)
        plt.figure()
        nx.draw(G, with_labels=True)
        plt.savefig(self.directory + "trimed_topology", format="PNG")
        self.logger.info(f"after elimination the Graph has edges {len(list(G.edges))}, {list(G.edges)}")
        return G