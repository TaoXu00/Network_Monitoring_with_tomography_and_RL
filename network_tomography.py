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

    def nt_engine(self, G, path_list):
        #disabled the find optimal prob_path
        path_matrix = self.construct_matrix(G, path_list)
        #self.logger.debug("Original Probing Path list: %s" %(path_list))
        #self.logger.debug("Original Probing Path matrix: %s" % (path_matrix))
        n_links_origin=0
        for path in path_list:
            n_links_origin+=len(path)
        # step1: Get any basis of the instantly selected shortest path among each pair of monitors
        probing_paths = self.find_any_basis_prob_paths(path_matrix)
        #step2: collect the end-to-end measurements
        b=self.end_to_end_measurement(G,probing_paths)
        #step3: Solve the system with Guassian Back propogation, x stores the solved links
        upper_triangular = self.triangular_matrix(G, probing_paths, b)
        x, count=self.back_substitution(upper_triangular)   #now got the identifiable links with its value
        #step4: estimate the bound for every unidentifiable link with the value calculated in x
        link_tight_bound_dict=self.calculate_bound_of_unidenticable_links(G, probing_paths, b, x)
        #self.logger.debug("link_tight_bound_dict %s" %(link_tight_bound_dict))
        n_links_optimal_path=np.count_nonzero(probing_paths)
        #step 5: pick the average of the tight bound and assign to the unidentificable links
        #print("before assign the bound: %s" %(x))
        for key in link_tight_bound_dict.keys():
            x[key]=link_tight_bound_dict[key]/2
        #print("after assignment of the bound %s" %(x))
        #step6: assgin the average of the probed links to the uncovered links
        n_probed_links=np.count_nonzero(x)
        avg=sum(x)/n_probed_links
        for i in range(len(x)):
            if x[i] ==0 or x[i]<0:
                x[i]=avg
        #print("after the assignment of the average to the uncovered links: %s" %(x))
        return x, count, n_links_origin,n_links_optimal_path

    def calculate_bound_of_unidenticable_links(self, G, probing_paths, b, x):
        solved_link_ids = np.nonzero(np.array(x))
        #self.logger.debug("set_solved_link_ids %s:" % (solved_link_ids))
        unindent_links_bound_dic={}
        for i in range(len(probing_paths)):
            link_ids_in_probing_path=np.nonzero(np.array(probing_paths[i]))
            solved_link_ids_in_current_path=np.intersect1d(solved_link_ids, link_ids_in_probing_path)
            sum=0
            for link_id in solved_link_ids_in_current_path:
                sum+=x[link_id]
            upper_bound=b[0][i]-sum
            unindent_links=np.setdiff1d(np.array(link_ids_in_probing_path), solved_link_ids_in_current_path)
            for free_link_id in unindent_links:
                if free_link_id in unindent_links_bound_dic.keys():
                    unindent_links_bound_dic[free_link_id].append(upper_bound)
                else:
                    upper_bound_list=[]
                    upper_bound_list.append(upper_bound)
                    unindent_links_bound_dic[free_link_id]=upper_bound_list
        #find the smallest bound
        link_tight_bound_dict={}
        for key in unindent_links_bound_dic.keys():
            bound_list=unindent_links_bound_dic[key]
            min_bound_list=min(bound_list)
            link_tight_bound_dict[key]=min_bound_list

        sorted_links = sorted(link_tight_bound_dict.items(), key=lambda x: x[1], reverse=True)
        sorted_link_tight_bound_decending_dict = dict(sorted_links)
        #self.logger.debug(sorted_link_tight_bound_decending_dict)
        link_id_decending = list(sorted_link_tight_bound_decending_dict.keys())
        # self.logger.debug("upper_triangular matrix: %s" %(upper_triangular))
        # step5: Re-arrange the LSM by the corresponding
        links_id_acending = list(sorted(link_tight_bound_dict.keys()))
        swap_dict = {}
        probing_paths=np.array(probing_paths)
        columns_of_unindentificable_links=probing_paths[:, link_id_decending]
        for i in range(len(links_id_acending)):
            # swapping the column of the matrix
            #dfdsfsf #check the swap part, here are some problems
            probing_paths[:, links_id_acending[i]] = columns_of_unindentificable_links[:, i]
            swap_dict[links_id_acending[i]] = link_id_decending[i]
        # step6: find the reduced row echelon form of the re-ordered variables
        upper_triangular = self.triangular_matrix(G, probing_paths, b)
        # find the range of the pivot
        for i in range(len(upper_triangular)):
            links_in_the_path = np.nonzero(upper_triangular[i][:-1])
            if len(links_in_the_path[0]) >0:
                pivot = links_in_the_path[0][0]
                rest_variables = np.setdiff1d(links_in_the_path, pivot)
                sum = 0
                if pivot in swap_dict.keys():  # it is a unidentificable link
                    for link in rest_variables:
                        if link in swap_dict.keys():  # it is unidentificalble link
                            origin = swap_dict[link]
                            sum += upper_triangular[i][link] * link_tight_bound_dict[origin]
                        else:  # it is solved link
                            sum += upper_triangular[i][link] * x[link]
                    pivot_bound = b[0][i] - sum
                    pivot_TNB = link_tight_bound_dict[swap_dict[pivot]]
                    if pivot_bound < pivot_TNB and pivot_bound >0:
                        link_tight_bound_dict[swap_dict[pivot]] = pivot_bound
        return link_tight_bound_dict

    def end_to_end_measurement(self, G, path_list):
        path_delays = []
        average_edge_delay_list=[]
        for path in path_list:
            #print("path: %s" %(path))
            path_delay = 0
            for i in range(len(path)):
                if path[i]!= 0:
                    edge=list(G.edges)[i]
                    path_delay = path_delay + G[edge[0]][edge[1]]['delay']
            path_delays.append(path_delay)
            #average_edge_delay=path_delay/len(path)
            #average_edge_delay_list.append(average_edge_delay)
        b = np.array([path_delays])  # the delay of the selected path
        return b

    def find_any_basis_prob_paths(self, path_matrix):
        #shuffle the row randomly
        path_matrix=np.array(path_matrix)
        np.random.shuffle(path_matrix)
        probing_paths=[]
        #self.logger.debug("shuffled_paths: %s" % (path_matrix))
        #self.logger.debug("after sort: %s" % (num_links_after_sort))
        for path in path_matrix:
            #m_optimal_probing_path=[]
            #self.logger.debug("#optimal_probing_paths1: %d" % (len(optimal_probing_paths)))
            #m_optimal_probing_path =optimal_probing_paths
            #self.logger.debug("copy_probing_paths1: %s" % (m_optimal_probing_path))
            rank1=np.linalg.matrix_rank(probing_paths)
            probing_paths.append(path)
            rank2=np.linalg.matrix_rank(probing_paths)
            #self.logger.debug("compare_probing_paths1: %d" % (len(optimal_probing_paths)))
            #self.logger.debug("rank1 %d, rank2: %d" % (rank1, rank2))
            if rank2 <= rank1:
                probing_paths=probing_paths[:-1]
                #self.logger.debug("deleting the added row")
                #self.logger.debug("%d paths in the optimal_probing path" %(len(optimal_probing_paths)))
        return probing_paths



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

    def triangular_matrix(self,G,path_matrix, b):
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
            #self.logger.debug("A(path): rank is: %d" %(rank))
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
            #m_rref, inds = sympy.Matrix(M).rref(iszerofunc=lambda x: abs(x) < 1e-12)
            #self.logger.debug("%d pivots(basic variables), their index are: %s}" %(len(list(inds)),list(inds)))
            uninds=list(range(len(path_matrix[0])))
            '''
            for i in inds:
                if i in uninds:
                    uninds.remove(i)
            '''
        #self.logger.info("%d free variables,the indexes are: %s " %(len(uninds), uninds))
        #self.logger.debug(f"the m_rref {m_rref}")
        #return triangular_matrix, list(inds), uninds
        return triangular_matrix

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
        #self.logger.debug("after deletion, the final inds are %s " %(inds))
        x=[0]*(n-1)
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
        if num_one==1:
            for i in range(len(last_row)-1):
                if last_row[i]==1:
                    x[i]=last_row[n-1]
                    #self.logger.debug(f"computed i {i} is {x[i]}")
                    break
        #elif num_one > 1:
        #    self.logger.debug(f"no edges has been computed in the last row")
        for i in range(total_row-2,-1,-1):
            #self.logger.debug(f"row i= {i}")
            for j in range(n-1):
                found_pivot = False
                if(upper_triangular_after_deletion[i][j]==1):
                    pivot=j
                    found_pivot=True
                    break
            if found_pivot == False:
                continue
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
            index=0
            deleted= False
            for row in sys:
                for j in uninds:
                    if row[j]!=0 or np.count_nonzero(row)==0:
                        sys=np.delete(sys,index,0)
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