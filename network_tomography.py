import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg
import sympy
np.set_printoptions(threshold=np.inf)
from numpy.random import seed
from numpy.random import randint
from numpy import random
import matplotlib.pyplot as plt
import math

from sympy import nsimplify
from sympy.matrices import Matrix
class network_tomography:
    def __init__(self, logger):
        self.logger=logger

    def nt_engine(self,G, path_list,b):
        path_matrix=self.construct_matrix(G, path_list)
        print(f"path_matrix:{path_matrix}")
        print(f"b={b}")

        M, inds, uninds = self.find_basis(G, path_matrix, b)

        x, count = self.edge_delay_infercement(G, M, inds, uninds)
        return x,count

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
            #rref=[[]]
            M=[[]]
        else:
            A=np.array(path_matrix,dtype=float)
            rank = np.linalg.matrix_rank(A)
            self.logger.debug(f"A(path): rank is: {rank}")
            #Matrix(b).applyfunc(nsimplify)
            M=np.concatenate((A,b.T),axis=1)
            self.logger.debug(M)
            rank = np.linalg.matrix_rank(M)
            self.logger.debug(f"M(path combined with measurement): rank is: {rank}")
            triangular_matrix = self.upper_triangular(M)
            self.logger.debug(f"triangular matrix:{np.array(triangular_matrix)}")
            m_rref, inds = sympy.Matrix(M).rref(iszerofunc=lambda x: abs(x) < 1e-12)
            #m_rref, inds = sympy.Matrix(M).rref()
            #(rm, cm) = m_rref.shape
            #self.logger.debug(f"m_rref: rm = {rm}, cm = {cm}")
            #self.logger.debug(f"the reduced matrix combined with the original measurements is: \n {m_rref}")
            self.logger.debug(f"{len(list(inds))} pivots(basic variables), their index are: {list(inds)}")

            uninds=list(range(len(path_matrix[0])))
            for i in inds:
                if i in uninds:
                    uninds.remove(i)
        self.logger.info(f"{len(uninds)} free variables,the indexes are: {uninds}")
        #self.logger.debug(f"the m_rref {m_rref}")
        return triangular_matrix, list(inds), uninds

    def end_to_end_measurement(self, G, path_list):
        print(f"pathlist= {path_list}")
        path_delays = []
        for path in path_list:
            print(f"path: {path}")
            path_delay = 0
            for edge in path:
                path_delay = path_delay + G[edge[0]][edge[1]]['delay_mean']
            path_delays.append(path_delay)
        b = np.array([path_delays])  # the delay of the selected path
            # print(f"end to end measures {b}")
            # b=np.array([[1.5,3.5,3.5,4,4,2]])
        return b

    def upper_triangular(self,M):
        # move all zeros to buttom of matrix
        self.logger.debug(f"M:{M}")
        M = np.concatenate((M[np.any(M != 0, axis=1)], M[np.all(M == 0, axis=1)]), axis=0)

        # iterate over matrix rows
        #for i in range(0, M.shape[0]):
        i=0
        k=0
        while i < M.shape[0] and k<M.shape[1]-1:
            # initialize row-swap iterator
            j = 1
            # select pivot value
            pivot = M[i][k]
            # find next non-zero leading coefficient
            while pivot == 0 and i + j < M.shape[0]:
                # perform row swap operation
                M[[i, i + j]] = M[[i + j, i]]
                # incrememnt row-swap iterator
                j += 1
                # get new pivot
                pivot = M[i][k]

            # if pivot is zero, check the next element in row i
            if pivot == 0:
                k +=1
            # extract row
            elif pivot!=0:
                row = M[i]
                # get 1 along the diagonal
                M[i] = row / pivot
                # iterate over remaining rows
                for j in range(i + 1, M.shape[0]):
                    # subtract current row from remaining rows
                    M[j] = M[j] - M[i] * M[j][k]
                i+=1
                k+=1
            #self.logger.debug(f"i={i}, M={M}")
        # return upper triangular matrix
        return M

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
            self.logger.debug(f"after eliminating the free variable, the solvable matrix is: \n{sys} ")
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
        for i in range(0,len(x[0])):
            if x[0][i]<=10**(-5):
                x[0][i]=0

        self.logger.debug(f"x= {x}")
        count=np.count_nonzero(x)
        self.logger.info(f"{count} edges are computed")

        return x,count