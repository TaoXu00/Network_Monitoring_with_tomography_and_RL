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

from sympy import nsimplify
from sympy.matrices import Matrix
class network_tomography:
    def end_to_end_measurement(self, G, path_list):
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
    def construct_matrix(self, G, path_list):
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
        #print(f"path matrix {path_matrix}")
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
    def find_basis(self,G,path_matrix, b):
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
            A=np.array(path_matrix,dtype=float)
            rank = np.linalg.matrix_rank(A)
            print(f"A(path): rank is: {rank}")
            #Matrix(b).applyfunc(nsimplify)
            M=np.concatenate((A,b.T),axis=1)
            print(M)
            rank = np.linalg.matrix_rank(M)
            print(f"M(path combined with measurement): rank is: {rank}")
            m_rref, inds = sympy.Matrix(M).rref(iszerofunc=lambda x: abs(x) < 10**-10)
            #m_rref, inds = sympy.Matrix(M).rref()
            (rm, cm) = m_rref.shape
            #print(f"m_rref: rm = {rm}, cm = {cm}")
            #print(f"the reduced matrix combined with the original measurements is: \n {m_rref}")
            #print(f"the pivots(basic variables) index are: {list(inds)}")

            uninds=list(range(len(path_matrix[0])))
            for i in inds:
                if i in uninds:
                    uninds.remove(i)
        print(f"the free variables indexes are: {uninds}")
        #print(f"the m_rref {m_rref}")
        return m_rref, list(inds), uninds
    '''
    Solve the system with the basic variables found in function find_basis()
    '''
    def edge_delay_infercement(self,G, m_rref, inds, uninds):
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
            #print('Type of A: {}'.format(type(A)))
            #print('inverse of A: {}'.format(np.linalg.pinv(A)))
            x=np.linalg.pinv(A).dot(b).T
            #print('precision: {}'.format(np.finfo(np.longdouble).precision))
            #x=np.round(x[0])
            #xf=list(np.round(x[0]))
        #print(f"the delay inference is: {x[0]}")
        count=0

        '''
        for i in x:
            if i>=0.9:
                #print(i)
                count = count + 1
        '''
        count=np.count_nonzero(x)
        print(f"{count} edges are computed")
        return x,count