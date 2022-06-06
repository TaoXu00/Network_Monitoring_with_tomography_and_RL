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

    def __init__(self, time):
        '''
        :param time: the total duration for MAB algorithm
        '''
        self.time=time
        self.Dict_edge_scales={}
        self.Dict_edge_delay_sample={}

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
        G=nx.erdos_renyi_graph(n, p)
        print("Graph Created! ")
        #seed(1)

        #Topology 5, fixed graph for multi-armed-bandits algorithm

        #G.add_weighted_edges_from([(0, 1, 1), (0, 2, 1), (0, 3, 1), (1, 4, 1),(1,2,1),(2,5,1), (2, 3, 1),(3,6,1), (4,7,1),(4,5,1),(5,6,1),(5,8,1),(6,9,1),(7,10,1),
        #                           (7,8,1), (8,10,1),(8,9,1), (9,10,1)])

        for edge in G.edges:
            #G[edge[0]][edge[1]]['delay']=randint(1,10)
            G[edge[0]][edge[1]]['weight']=1
        print("all the edge weights in the graph are assigned to 1")
        self.construct_link_delay_distribution(G)
        print(f"Edge delay scales: {self.Dict_edge_scales}")
        self.draw_edge_delay_sample(G)
        self.assign_link_delay(G)
        #show the topology graph
        nx.draw(G, with_labels=True)
        plt.show()
        #graphy=plt.subplot(122)
        #nx.draw(G,pos=nx.circular_layout(G),node_color='r', edge_color='b')

        print(G.name)
        print(f"Graph Nodes: {G.nodes}")
        print(f"Graph Edges length:{len(G.edges)}\nGraph Edges data: {G.edges.data()}")
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
        scales=np.random.randint(1, 20, len(G.edges))

        #scales=np.array([18, 3, 9 ,10, 9, 15, 18, 13, 15, 18, 4, 12, 7, 2, 16, 6, 6, 5])
        i=0
        for edge in G.edges:
           self.Dict_edge_scales[edge]=scales[i]
           G[edge[0]][edge[1]]['delay-mean']=scales[i]
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
        print(f"Draw {self.time} delay examples from exponential distribution for each edge.")
        average = [np.average(self.Dict_edge_delay_sample[edge]) for edge in G.edges]
        print(f"edge delay sample average {average}")


    def assign_link_delay(self,G):
        '''
        for edge in G.edges:
             G[edge[0]][edge[1]]['delay']=1
        '''
        for edge in G.edges:
            #print(f"Dict_edge_delay_sample: {edge} {self.Dict_edge_delay_sample[edge]}")
            G[edge[0]][edge[1]]['delay'] = self.Dict_edge_delay_sample[edge][0]
            #print(f"type: {type(self.Dict_edge_delay_sample[edge])}")
            self.Dict_edge_delay_sample[edge]=np.delete(self.Dict_edge_delay_sample[edge],0)
            #print(f"after deletion{self.Dict_edge_delay_sample[edge]}")
        print(f"Assigned Delay {G.edges.data()}")
        #####to do  remove the first element and run it again.

        # print(f"updated the edge delay: {G.edges.data()}")

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
        print(G.nodes)
        monitors = []
        print(f"n={n} monitor_candidate_list={len(monitor_candidate_list)}")
        if len(monitor_candidate_list) == n:
            monitors = monitor_candidate_list
        elif len(monitor_candidate_list) < n:
            monitors = monitor_candidate_list
            rest_nodes = [elem for elem in G.nodes if elem not in monitors]
            select = sample(rest_nodes, k=n - len(monitor_candidate_list))
            monitors = monitors + select
        print(f"Monitors are deployed in nodes: {monitors}")
        return monitors

    def getPath(self,G, monitors):
        nodepairs = [(monitors[i], monitors[j]) for i in range(len(monitors)) for j in range(i + 1, len(monitors))]
        # print(f"end to end nodepairs: {nodepairs}")
        path_list = []
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

    def trimNetwrok(self, G, monitors):
        '''
        this function aims to eliminate all the links that are not covered in any simple path between any pair of monitors
        :param G: the original topology
        :return: the trimed topology
        '''
        print("Trim the network...............")
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
        #print(f"end to end paths set: {all_paths_set}")
        edges=list(G.edges)
        uncovered_edges=[]
        for edge_e in edges:
                edge_r=(edge_e[1],edge_e[0])
                if edge_e not in all_paths_set and edge_r not in all_paths_set:
                    #print(f"1 {edge_e}")
                    uncovered_edges.append(edge_e)

        print(f"uncovered edges: {uncovered_edges}")
        old_G=G.copy()   #store the original network topology
        for edge in uncovered_edges:
            G.remove_edge(*edge)
        print(f"after elimination the Graph has edges {len(list(G.edges))}, {list(G.edges)}")
        return G




