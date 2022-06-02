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

def graph_Generator(n,p):
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
    #G=nx.erdos_renyi_graph(n,p)
    #seed(1)

    #Topology 5, fixed graph for multi-armed-bandits algorithm
    G.add_weighted_edges_from([(0, 1, 1), (0, 2, 1), (0, 4, 1), (0, 7, 1), (0, 8, 1), (0, 9, 1), (1, 4, 1), (1, 5, 1), (1, 6, 1), (1, 8, 1), (
    1, 9, 1), (2, 3, 1), (2, 4, 1), (2, 7, 1), (3, 4, 1), (3, 5, 1), (3, 6, 1), (3, 7, 1), (4, 5, 1), (4, 9, 1), (
    5, 6, 1), (5, 7, 1), (5, 9, 1), (6, 9, 1), (7, 8, 1)])

    '''
    for edge in G.edges:
        #G[edge[0]][edge[1]]['delay']=randint(1,10)
        G[edge[0]][edge[1]]['weight']=1
    '''
    Dict_edge_scales=construct_link_delay_distribution(G)
    #print(Dict_edge_scales)
    assign_link_delay(G, Dict_edge_scales)
    nx.draw(G, with_labels=True)
    plt.show()
    #graphy=plt.subplot(122)
    #nx.draw(G,pos=nx.circular_layout(G),node_color='r', edge_color='b')

    print(G.name)
    print(f"Graph Nodes: {G.nodes}")
    print(f"Graph Edges length:{len(G.edges)}\nGraph Edges data: {G.edges.data()}")
    #print(nx.to_numpy_matrix(G,nodelist=['A','B','C','D'])) #get the adjacent matrix of the graph
    return G,Dict_edge_scales


def construct_link_delay_distribution(G):
    '''
    -The link delay distribution is defined as exponential model: f(x,λ)=λexp(-λx)
    1/λ is the mean of random variable x.the scale parameter in the exponential function is 1/λ. suppose every link delay
    is independent and all of them follow the exponential distribution with different parameter λ.
    Let's set the minimum link delay is 1 and the maximum link delay is 5, then we randomly select #G.edge numbers in [1,6) as
    the 1/λ vector(scale vector) of size #G.size, then we will construct #G.edge exponential distributions for each link.
    :param G: the generated graph
    :return: the scales vector for all the edge
    '''
    scales=np.random.randint(1, 6, len(G.edges))
    Dict_edge_scales={}
    i=0
    for edge in G.edges:
       Dict_edge_scales[edge]=scales[i]
       G[edge[0]][edge[1]]['delay-mean']=scales[i]
       i=i+1
    print(f"scales:{scales}")
    return Dict_edge_scales

def assign_link_delay(G,Dict_edge_scales):
    for edge in G.edges:
        G[edge[0]][edge[1]]['delay'] = np.random.exponential(scale=Dict_edge_scales[edge],size=1)[0]
    #print(f"updated the edge delay: {G.edges.data()}")

def deploy_monitor(G,n,monitor_candidate_list):
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
    #monitors for topology 2
    #monitors=['A','B','E','F']  #all the end nodes are selected to deploy the monitor

    #monitors for topology 3
    #monitors=['A','B','I','F']
    monitors=[]
    print(f"n={n} monitor_candidate_list={len(monitor_candidate_list)}")
    if len(monitor_candidate_list)==n:
        monitors=monitor_candidate_list
    elif len(monitor_candidate_list)<n:
        monitors=monitor_candidate_list
        rest_nodes=[elem for elem in G.nodes if elem not in monitors]
        select=random.sample(rest_nodes,k=n-len(monitor_candidate_list))
        monitors = monitors+select
    print(f"Monitors are deployed in nodes: {monitors}")
    return monitors

'''
Get the path of the graph
input: the graph, end node list
output: the list of the path between each pair of end nodes
'''
def getPath(G,monitors):
    nodepairs=[(monitors[i],monitors[j]) for i in range(len(monitors)) for j in range(i+1, len(monitors)) ]
    #print(f"end to end nodepairs: {nodepairs}")
    path_list= []
    for n1,n2 in nodepairs:
        shortest_path=nx.shortest_path(G,source=n1, target=n2, weight='delay', method='dijkstra')
        #print(f"shortest path: {shortest_path}")
        pathpair=[]
        [pathpair.append((shortest_path[i],shortest_path[i+1])) for i in range(len(shortest_path)-1)]
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

def end_to_end_measurement(G,path_list):
    path_delays=[]
    for path in path_list:
        #print(f"path: {path}")
        path_delay = 0
        for edge in path:
           path_delay = path_delay+ G[edge[0]][edge[1]]['delay']
        path_delays.append(path_delay)
    b=np.array([path_delays])  #the delay of the selected path
    #print(f"end to end measures {b}")
    #b=np.array([[1.5,3.5,3.5,4,4,2]])
    return b
'''
Construct the network as a linear system Ax=b
A: path matrix(p*l) p is the number of the path, l is the number of links, A[i][j]=1 if link(edge) j in the path i, other wise 0.
x: a row matrix of size l. l[i] represents the delay of link.
b: a column matrix of length length basis(A), it represents the end to end measurements
input: pathlist and the Graph
output: the matrix of the pathlist 
'''
def construct_matrix(G, path_list):
    edges=list(G.edges)
    #print(f"con: edges {edges}")
    dict={}
    '''
    for idx, edge in enumerate(sorted(edges)):
        dict[edge]=idx+1
        print(f"index is {idx} and edge is {edge}")
    print(dict)
    '''
    path_matrix=[]
    for path in path_list:
        a=[]
        for edge_e in edges:
            edge_r=(edge_e[1],edge_e[0])
            if edge_e in path or edge_r in path:
                #print(f"1 {edge_e}")
                a.append(1)
            else:
                #print(f"0 {edge_r}")
                a.append(0)
        path_matrix.append(a)
    #(f"path matrix {path_matrix}")
    return path_matrix
'''
compute the basis of the path matrix, that is finding the reduced matrix wrt the measurement matrix b
Input:  
 ** path_matrix: the end-to-end measurement path_matrix P*E (P is the path number and E is the edges number)
 ** b, the measurement matrix
Output;
 ** m_rref:  the reduced matrix which is the concatenation of path_matrix and b, P*E
 ** inds: the indentificatable variables (list)
 ** unids: the unidentificatable variables (list)
'''
def find_basis(G,path_matrix, b):
    #a = np.array([[2., 4., 4., 4.], [1., 2., 3., 3.], [1., 2., 2., 2.], [1., 4., 3., 4.]])
    #contatenate path-matrix and b.
    (r,c)=b.shape
    #print(f"len(path_matrix) {len(path_matrix)} b= {b} r= {r} c={c} ")
    if len(path_matrix)==0 and c==0:
        rank=0
        inds=[]
        uninds=G.nodes
        m_rref=[[]]
    else:
        A=np.array(path_matrix)
        M=np.concatenate((A,b.T),axis=1)
        rank = np.linalg.matrix_rank(M)
        print(f"rank is: {rank}")
        #m_rref, inds = sympy.Matrix(path_matrix).T.rref()
        m_rref, inds= sympy.Matrix(M).rref()
        #print(f"the reduced matrix combined with the original measurements is: \n {m_rref}")
        print(f"the pivots(basic variables) index are: {list(inds)}")
        uninds=list(range(len(path_matrix[0])))
        for i in inds:
            if i in uninds:
                uninds.remove(i)
    print(f"the free variables indexes are: {uninds}")
    return m_rref, list(inds), uninds
'''
Solve the system with the basic variables found in function find_basis()
'''
def edge_delay_infercement(G,m_rref, inds, uninds):
    ## manipulate the matrix by removing the rows which contains free variables
    if len(inds) == 0:
        x=[0* len(G.edges)]
    else:
        sys=np.array(m_rref,dtype=np.float64)
        #print(sys)
        #print(f"the reduced matrix combined with the original measurements is: \n {sys}")
        #print(f"the pivots(basic variables) index are: {list(inds)}")
        #print(f"the free variables indexes are: {uninds}")
        index=0
        deleted= False
        for row in sys:
            for j in uninds:
                #print(f"checking row {index} at position {j}")
                #print(row)
                '''
                if(row[j]!=0):
                    sys[index]=0
                    break
                '''
                if row[j]!=0 or np.count_nonzero(row)==0:
                    sys=np.delete(sys,index,0)
                    #print(f"deleted row {index}")
                    #print(sys)
                    deleted = True
                    break
            if deleted == False:
                index=index+1
            else:
                deleted = False
        #print(f"after eliminating the free variable, the solvable matrix is: \n{sys} ")
        ## decompose the matrix into the form Ax=b, b is the last column and A is the [0:n-1] column
        (a,b)=sys.shape
        if(b>a):
            sys=np.pad(sys,((0,b-(a+1)),(0,0)), mode='constant', constant_values=float(0))
        else:
            diff=a-b
            for i in range(diff):
                np.delete(sys,a-1)
                a=a-1
        #print(f"square: {sys_square}")
        (a,b)=sys.shape
        A,b=np.hsplit(sys,[b-1])
        #print(A)
        #print(b)
        x=np.linalg.pinv(A).dot(b).T
        x=np.round(x[0])
        #xf=list(np.round(x[0]))
    print(f"the delay inference is: {x}")
    count=0
    for i in x:
        if i>=0.9:
            #print(i)
            count = count + 1
    #count=np.count_nonzero(x)
    print(f"{count} edges are computed")
    return x,count

def workflow(G, n, monitor_candidate_list):
    monitors=deploy_monitor(G,n, monitor_candidate_list)
    path_list=getPath(G,monitors)
    path_matrix=construct_matrix(G,path_list)
    b=end_to_end_measurement(G,path_list)
    m_rref, inds, uninds=find_basis(G,path_matrix,b)
    x,count=edge_delay_infercement(G,m_rref,inds,uninds)
    return monitors,x,count


def main(num_nodes,prob):
    G = graph_Generator(num_nodes,prob)
    monitors_list=[]
    monitors=[]
    monitors_count=[]
    solved_edges=[]
    solved_edges_count=[]
    for n in range(0,num_nodes+1,5):
        monitors,x,count = workflow(G, n, monitors)
        #print(f"n={n},monitors={monitors}")
        monitors_list.append(monitors)
        #print(f"append monitors_list:{monitors_list}")
        solved_edges.append(x)
        solved_edges_count.append(count)
        #print(f"append solved_edges_count:{count}")
    #print(monitors_list)
    #print(solved_edges_count)
    x=[len(monitors)/len(G.nodes) for monitors in monitors_list]
    y=[edges_count/len(G.edges) for edges_count in solved_edges_count]
    print(x,y)
    plt.plot(x,y)
    plt.xlabel("% of nodes selected as monitors")
    plt.ylabel("% of solved edges")
    #plt.show()
    plt.savefig('network_tomography.png')

#main(100, 0.5)
def Initialize(G,source, destination, Dict_edge_scales, optimal_delay):
    '''
    :param G: The network topology
    :param source: the source node
    :param destination:  the destination node
    :param Dict_edge_scales: the vector used to construct the delay exponential distribution
    :param optimal_delay: the delay computed with mean vector
    :return: Dict_edge_theta: updated real mean delay vector , Dict_edge_m: updated vector for tracking how many times this link has been visited so far,
             t - the timestamp, total_rewards - accumulate rewards, total_regrets -accumulate regrets
    '''

    #find the shortest path between 3 and 8
    num = len(list(nx.all_simple_paths(G, source=source, target=destination)))
    print(f"the total path number is {num}")
    #maintain t,wo vectors (1*N(#of edge)), theta and m. theta is the average(sample mean) of all the observed values of Xi up to the
    #current time-slot, m is the number of times that Xi has been observed up to the current time-slot.
    Dict_edge_theta={}
    Dict_edge_m={}
    for edge in G.edges:
        Dict_edge_theta[edge]=0
    for edge in G.edges:
        Dict_edge_m[edge]=0

    counter = 0
    t=1
    total_rewards=[]
    total_regrets=[]
    sample_delays=[]
    while(counter!=len(G.edges)):
        '''store the sampled delay for each link'''
        delays=[]
        for edge in G.edges:
            delays.append(G[edge[0]][edge[1]]['delay'])
        #print(delays)
        sample_delays.append(delays)
         #index of time slot
        shortest_path = nx.shortest_path(G, source=source, target=destination, weight='weight', method='dijkstra')
        #print(f" t = {t}, shortest path: {shortest_path}")
        # observe the links in the returned shortest path and update the theta vector and m vector
        total_weight = 0
        pathpair = []
        for i in range(len(shortest_path) - 1):
            if (shortest_path[i], shortest_path[i + 1]) in Dict_edge_theta:
                pathpair.append((shortest_path[i], shortest_path[i + 1]))
            else:
                pathpair.append((shortest_path[i + 1], shortest_path[i]))
        #if shortest_path not in shortest_path_list:
        #    print("Detect a new shortest path")
        #    rewards=0
        #    shortest_path_list.append(shortest_path)
        rewards=0
        for edge in pathpair:
            #print(f"edge {edge}")
            Dict_edge_theta[edge] = (Dict_edge_theta[edge] + G[edge[0]][edge[1]]['delay']) / (Dict_edge_m[edge] + 1)
            Dict_edge_m[edge] = Dict_edge_m[edge] + 1
            total_weight += G[edge[0]][edge[1]]['weight']
            G[edge[0]][edge[1]]['weight'] +=1
            G.edges.data()
            rewards+=G[edge[0]][edge[1]]['delay']
        #print(f"Dict_edge_m: {Dict_edge_m}")
        #print(f"rewards: {rewards}")
        total_rewards.append(rewards)
        #print(f"total_rewards: {total_rewards}")
        regret=sum(total_rewards)-t*optimal_delay
        total_regrets.append(regret)
        #print(f"regret: {regret}")
        #print(f"total_regrets: {total_regrets}")
        #print(Dict_edge_theta)
        #print(Dict_edge_m)
        assign_link_delay(G,Dict_edge_scales)
        t = t + 1
        counter=0
        for item in Dict_edge_m.values():
            if item!=0:
                counter=counter+1
    print("===============================Initialization is finished======================================================== ")
    return Dict_edge_theta, Dict_edge_m, t, total_rewards, total_regrets

def optimal_path(G, source, destination):
    optimal_delay =0
    shortest_path = nx.shortest_path(G, source=source, target=destination, weight='delay-mean', method='dijkstra')
    print(f"optimal path: {shortest_path}")
    for i in range(len(shortest_path)-1):
        optimal_delay+=G[shortest_path[i]][shortest_path[i+1]]["delay-mean"]
    print(f"optimal_delay: {optimal_delay}")
    return optimal_delay


def LLC_policy(G, Dict_edge_theta, Dict_edge_m, t, source, destination, total_rewards, offset):
    # select a path which solves the minimization problem
    for edge in G.edges:
        #llc_factor=Dict_edge_theta[edge] + math.sqrt((len(G.edges) + 1) * math.log(t) / Dict_edge_m[edge])
        #print(f"llc_factor: {llc_factor}")
        G[edge[0]][edge[1]]["llc_factor"]=Dict_edge_theta[edge]-math.sqrt((len(G.edges)+1)*math.log(t)/Dict_edge_m[edge])+offset
    #select the shortest path with wrt the llc_fact
    shortest_path = nx.shortest_path(G, source=source, target=destination, weight='llc_factor', method='dijkstra')
    print(f"shortest path: {shortest_path}")
    #print(G.edges.data())
    #update the Dict_edge_theta and Dict_edge_m
    pathpair=[]
    rewards=0
    for i in range(len(shortest_path) - 1):
        if (shortest_path[i], shortest_path[i + 1]) in Dict_edge_theta:
            pathpair.append((shortest_path[i], shortest_path[i + 1]))
        else:
            pathpair.append((shortest_path[i + 1], shortest_path[i]))
    for edge in pathpair:
        print(f"edge {edge}")
        print(f"Dict_edge_theta[edge] {Dict_edge_theta[edge]} +  G[edge[0]][edge[1]]['delay']: {G[edge[0]][edge[1]]['delay']}")
        print(f"Dict_edge_m[edge] {Dict_edge_m[edge]}")
        Dict_edge_theta[edge] = (Dict_edge_theta[edge] + G[edge[0]][edge[1]]['delay']) / (Dict_edge_m[edge] + 1)
        Dict_edge_m[edge] = Dict_edge_m[edge] + 1
        rewards=rewards+G[edge[0]][edge[1]]['delay']
    print(f"Dict_edge_m: {Dict_edge_m}")
    print(f"rewards:{rewards}")
    total_rewards.append(rewards)
    #print(f"total_rewards:{total_rewards}")
    print(Dict_edge_theta)
    return total_rewards

def train_llc(G,source, destination, round, Dict_edge_theta, Dict_edge_m,optimal_delay, total_rewards,t,Dict_edge_scales,total_regrets):
    offset= math.sqrt((len(G.edges)+1)*math.log(t+round))
    for i in range(round):
        print(f"t={t}")
        assign_link_delay(G,Dict_edge_scales)
        total_rewards= LLC_policy(G, Dict_edge_theta, Dict_edge_m, t,source, destination, total_rewards,offset)
        regret=sum(total_rewards)-t*optimal_delay
        #print(f"regretes:{regret}")
        total_regrets.append(regret)
        print(total_regrets)
        t = t + 1  # the time slot increase 1
    return total_regrets, t

def MAB():
    G, Dict_edge_scales = graph_Generator(10, 0.5)
    source = 3
    destination = 8
    optimal_delay=optimal_path(G,3,8)
    Dict_edge_theta, Dict_edge_m, t, total_rewards, total_regrets= Initialize(G,source, destination, Dict_edge_scales,optimal_delay)
    #print(f"t={t}, len_total_reward: {len(total_rewards)}")
    #total_regrets,t=train_llc(G,source, destination, 50, Dict_edge_theta, Dict_edge_m,optimal_delay, total_rewards,t,Dict_edge_scales,total_regrets)
    #print(f"len total_regrets: {len(total_regrets)}")
    avg_total_regrets=[ total_regrets[i]/(i+1) for i in range(len(total_regrets))]
    #x=[i+1 for i in range(t-1)]
    #print(x)
    '''
    plt.plot(x,avg_total_regrets)
    plt.xlabel("time slot")
    plt.ylabel("avg_regrets")
    '''
    #plt.plot(x, total_regrets)
    #plt.xlabel("time slot")
    #plt.ylabel("total_regrets")
    #plt.show()
    #plt.savefig('network_tomography.png')


MAB()






