import numpy as np
import plotter as plotter
import os
import matplotlib.pyplot as plt


def get_CDF_link_utility(dir, num_edge):
    files = [file for file in os.listdir(dir) if not file.startswith('.')]
    #divide the bin to [0,1,0.1], 0 means the link is not shared by any path, 1 means the link is shared by all the paths.
    dict_link_utility={}
    num_path=[10, 45, 105, 190, 300]
    for file in files: # each file
        link_utility_2D_list=np.loadtxt(dir+file)
        for j in range(0, 5): #each row that is one number of monitor percentage
            #initialize the bin and the rate =0
            for i in range(0, 100, 10):
                key = (i, i + 10)
                dict_link_utility[key] = 0
            link_utility=link_utility_2D_list[j]
            link_utility_pert=link_utility/num_path[j]*100
            #print(link_utility_pert)
            for lr in link_utility_pert:
                index=int(lr/10)
                keys=list(dict_link_utility.keys())
                dict_link_utility[keys[index]]+=1
            for key in dict_link_utility.keys():
                dict_link_utility[key]=dict_link_utility[key]/num_edge
            #print(dict_link_utility)
            #do the plot using the dictionary.
            bins=['[0,0.1)', '[0.1, 0.2]', '[0.2,0.3)', '[0.3, 0.4)', '[0.4, 0.5)','[0.5,0.6)', '[0.6, 0.7)', '[0.7,0.8)', '[0.8, 0.9)', '(0.9, 1.0]']
            values=list(dict_link_utility.values())
            # accums=[]
            # for i in range(1, len(values)+1):
            #     accum=np.sum(values[:i])
            #     accums=np.append(accums, accum)
            plt.figure()
            barWidth = 0.3
            bar = np.arange(len(values))
            plt.bar(bins, values,  width=barWidth)
            plt.xticks(rotation=45)
            plt.xlabel('% of path')
            plt.ylabel('% of link')
            plt.savefig('%s%s%s.png' %(dir,file,j),  bbox_inches="tight")

myplotter=plotter.plotter('Network_tomography/')
myplotter.plot_avg_e2e_delay_BR50_10_50_pert_monitors('avg_e2e_delay_BR_50_10_50_pert_monitors/', [10,20,30,40,50])
#get_CDF_link_utility('avg_link_utility_BR50_10_50_pert_monitors/', 96)
