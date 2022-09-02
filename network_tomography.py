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

    def nt_engine(self, G, path_list, b):
        path_matrix = self.construct_matrix(G, path_list)
        print("path_matrix:%s" %(path_matrix))
        print(f"b=%s" %(b))
        upper_triangular, inds, uninds = self.find_basis(G, path_matrix, b)
        x, count=self.back_substitution(upper_triangular)
        #x, count = self.edge_delay_infercement(G, M, inds, uninds)
        return x, count


    def end_to_end_measurement(self, G, path_list,weight):
        path_delays=[]
        for path in path_list:
            path_delay = 0
            for edge in path:
               path_delay = path_delay+ G[edge[0]][edge[1]][weight]
            path_delays.append(path_delay)
        b=np.array([path_delays])  #the delay of the selected path
        return b

    def construct_matrix(self, G, path_list):
        '''
        Construct the network as a linear system Ax=b
        A: path matrix(p*l) p is the number of the path, l is the number of links, A[i][j]=1 if link(edge) j in the path i, other wise 0.
        x: a row matrix of size l. l[i] represents the delay of link.
        b: a column matrix of length length basis(A), it represents the end to end measurements
        :param G: Graph G
        :param path_list:
        :return: the matrix of the pathlist with dimension (P*E) P is the path number, E is the link number
        '''
        edges=list(G.edges)
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

    def find_basis(self,G,path_matrix, b):
        '''
        compute the basis of the path matrix, that is finding the reduced matrix for the concatenate matrix of path matrix and measurement matrix b
        :param G:
        :param path_matrix: the end-to-end measurement path_matrix P*E (P is the path number and E is the edges number)
        :param b: the measurement matrix
        :return: m_rref:  the reduced matrix for the concatenation of path_matrix and b, P*E
                 inds:    the indentificatable variables (list)
                 unids:   the unidentificatable variables (list)
        '''
        (r,c)=b.shape
        if len(path_matrix)==0 and c==0:
            rank=0
            inds=[]
            uninds=G.nodes
            triangular_matrix=[[]]
        else:
            A=np.array(path_matrix,dtype=float)
            rank = np.linalg.matrix_rank(A)
            self.logger.debug("A(path): rank is: %d" %(rank))
            #Matrix(b).applyfunc(nsimplify)
            M=np.concatenate((A,b.T),axis=1)
            rank = np.linalg.matrix_rank(M)
            #self.logger.debug("M(path combined with measurement): rank is: %d" %(rank))
            triangular_matrix = self.upper_triangular(M)
            for row in triangular_matrix:
                for i in range(len(row)):
                    if row[i]==1:
                        #self.logger.debug(f"finding pivots in triangular matrix:")
                        #self.logger.debug(f"i: {i}")
                        break
            #self.logger.debug(f"triangular matrix:{np.array(triangular_matrix)}")
            m_rref, inds = sympy.Matrix(M).rref(iszerofunc=lambda x: abs(x) < 1e-12)
            self.logger.debug("%d pivots(basic variables), their index are: %s}" %(len(list(inds)),list(inds)))
            uninds=list(range(len(path_matrix[0])))
            for i in inds:
                if i in uninds:
                    uninds.remove(i)
        self.logger.info("%d free variables,the indexes are: %s " %(len(uninds), uninds))
        #self.logger.debug(f"the m_rref {m_rref}")
        return triangular_matrix, list(inds), uninds

    def upper_triangular(self,M):
        '''
        Transter the Matrix M as an upper triangular matrix leveraging the Gaussian elimination process
        :param M: A matrix with dimension (P*E), column number is not necessary to be equal as row column
        :return:  M:  matrix in a upper triangular form
        '''
        # move all zeros to buttom of matrix
        #self.logger.debug(f"M:{M}")
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
    def back_substitution(self,upper_triangular):
        n=len(upper_triangular[0])
        print(f"n: {n}")
        delete=[]
        for i in range(len(upper_triangular)):
            if abs(upper_triangular[i][n-1])< 1e-12:
                #self.logger.debug(f"back_substitution: found {upper_triangular[i][n-1]}")
                delete.append(i)
        #self.logger.debug(f"deleted rows: {delete}")
        upper_triangular_after_deletion=np.delete(upper_triangular, delete, 0)
        #self.logger.debug(f"after deletion upper_triangular: {upper_triangular_after_deletion}")
        inds = []
        for row in upper_triangular_after_deletion:
            for i in range(n-1):
                if row[i] ==1:
                    inds.append(i)
                    break
        self.logger.debug("after deletion, the final inds are %s " %(inds))
        x=[0]*(n-1)
        #print(f"len of x: {len(x)}")
        #self.logger.debug(f"inds after deletion: {inds}")
        ''''
        for i in inds:
            if i<n-1:
                print(f"i={i}")
                x[i]=-1
        '''
        #self.logger.debug(f"x: before_back_subsituation: {x}")
        total_row=len(upper_triangular_after_deletion)
        last_row= upper_triangular_after_deletion[len(upper_triangular_after_deletion)-1]
        ##modify here if the last row has more than one '1', there is no solution
        last_row_path=last_row[:(n-1)]
        num_one=np.count_nonzero(last_row_path==1)
        if num_one >1:
            self.logger.debug(f"no edges has been computed in the last row")
        elif num_one==1:
            for i in range(len(last_row)-1):
                if last_row[i]==1:
                    x[i]=last_row[n-1]
                    #self.logger.debug(f"computed i {i} is {x[i]}")
                    break
        for i in range(total_row-2,-1,-1):
            #self.logger.debug(f"row i= {i}")
            for j in range(n-1):
                if(upper_triangular_after_deletion[i][j]==1):
                    pivot=j
                    break
            x[pivot] = upper_triangular_after_deletion[i][n - 1]
            #self.logger.debug(f"pivot= {j}, sum ={x[pivot]}")
            for k in range(pivot+1, n-1):
                if upper_triangular_after_deletion[i][k]!=0 and x[k] == 0:
                    #self.logger.debug(f"k={k}")
                    #self.logger.debug(f"x[{pivot}] is unidentifiable")
                    x[pivot]=0
                    break
                else:
                    x[pivot]=x[pivot]-upper_triangular_after_deletion[i][k]*x[k]
            #self.logger.debug(f"computed i {i} is {x[pivot]}")
        count = np.count_nonzero(x)
        #self.logger.info(f"{count} edges are computed")
        return x, count


    def edge_delay_infercement(self,G, m_rref, inds, uninds):
        '''
        Solve the system with the identified variables
        :param G: Graph G
        :param m_rref: a Matrix of the upper triangular form
        :param inds: the identified variables
        :param uninds: unidentificable variables
        :return: x   :the matrix of the solved links
                count: the number of the solved links
        '''
        ## manipulate the matrix by removing the rows which contains free variables
        if len(inds) == 0:
            x=[[0* len(G.edges)]]
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
            #self.logger.debug(f"after eliminating the free variable, the solvable matrix is: \n{sys} ")
            ## decompose the matrix into the form Ax=b, b is the last column and A is the [0:n-1] column
            #self.logger.debug(f"after eliminating free variables: {sys}")
            x_back = self.back_substitution(sys)
            #self.logger.debug(f"x_back:{x_back}")
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
            x=np.linalg.pinv(A).dot(b).T
            
        for i in range(0,len(x[0])):
            if x[0][i]<=10**(-5):
                x[0][i]=0
        self.logger.debug("x= %s" %(x))
        count=np.count_nonzero(x)
        self.logger.info("%d edges are computed" %(count))
        return x,count