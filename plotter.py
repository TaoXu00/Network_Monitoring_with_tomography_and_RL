import matplotlib.pyplot as plt
import numpy as np

class plotter:
    def __init__(self, directory):
        self.directory= directory

    def plot_edge_exploitation_times_bar_combined(self,edge_exploration_times):
        # set width of bar
        barWidth = 0.25
        fig = plt.subplots()
        # set height of bar
        init = edge_exploration_times[0]
        t3000 =edge_exploration_times[1]
        # Set position of bar on X axis
        br1 = np.arange(len(edge_exploration_times[0]))
        br2 = [x + barWidth for x in br1]

        # Make the plot
        plt.bar(br1, init, color='r', width=barWidth,
                edgecolor='grey', label='init')
        plt.bar(br2, t3000, color='g', width=barWidth,
                edgecolor='grey', label='t3000')

        # Adding Xticks
        xlable = [x for x in range(len(edge_exploration_times[0]))]
        plt.xlabel('linkID', fontweight='bold', fontsize=8)
        plt.ylabel('# of exploration ', fontweight='bold', fontsize=15)
        plt.xticks([r for r in range(len(edge_exploration_times[0]))], range(len(edge_exploration_times[0])))
        plt.legend()
        plt.savefig(self.directory + '# of edge exploration', format="PNG", dpi=300,
                    bbox_inches='tight')

    def plot_edge_exploitation_times_bar(self,label,Dict_edge_m):
        fig = plt.figure(figsize=(10, 7))
        plt.xlabel('linkID', fontweight='bold', fontsize=15)
        plt.ylabel('#times the edge is observed', fontweight='bold', fontsize=15)
        # Horizontal Bar Plot
        bar=np.arange(len(Dict_edge_m))
        plt.bar(bar,Dict_edge_m.values(), label= label)
        plt.xticks(np.arange(len(Dict_edge_m)))
        plt.legend()

    def plot_edge_delay_difference_alongtime(self, s,e, edge_delay_difference_list):
        # set width of bar
        barWidth = 0.25
        fig = plt.subplots()
        # set height of bar
        init= edge_delay_difference_list[0][s:e]
        t1000 = edge_delay_difference_list[1][s:e]
        t2000 = edge_delay_difference_list[2][s:e]
        t3000= edge_delay_difference_list[3][s:e]
        # Set position of bar on X axis
        br1 = np.arange(e-s)
        br2 = [x + barWidth for x in br1]
        br3 = [x + barWidth for x in br2]
        br4 = [x + barWidth for x in br3]
        # Make the plot
        plt.bar(br1, init, color='r', width=barWidth,
                edgecolor='grey', label='init')
        plt.bar(br2, t1000, color='g', width=barWidth,
                edgecolor='grey', label='t1000')
        plt.bar(br3, t2000, color='b', width=barWidth,
                edgecolor='grey', label='t2000')
        plt.bar(br4, t3000, color='c', width=barWidth,
                edgecolor='grey', label='t3000')

        # Adding Xticks
        xlable=[x for x in range(s,e)]
        plt.xlabel('linkID', fontweight='bold', fontsize=15)
        plt.ylabel('delay difference from the mean', fontweight='bold', fontsize=15)
        plt.xticks([r for r in range(e-s)], range(s,e,1))
        plt.legend()

    def plot_edge_delay_difference(self,G,Dict_edge_theta):
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        langs = list(range(1,len(G.edges)+1))
        print(f"langs:{langs}")
        delay_diff=[]
        for edge in G.edges:
            diff=abs(Dict_edge_theta[edge]-G[edge[0]][edge[1]]['delay_mean'])
            delay_diff.append(diff)
        ax.bar(langs, delay_diff)
        plt.xlabel("Link ID")
        plt.ylabel("delay difference from mean")

    def plot_total_edge_delay_mse(self,total_mse_array):
        # plot the mse
        plt.figure()
        x = range(len(total_mse_array))
        print(f"mse:{x}")
        print(f"mse:{total_mse_array}")
        plt.plot(x, total_mse_array)
        plt.xlabel("time slot")
        plt.ylabel("total_mse of all edges dalay")
        # plt.show()
        plt.savefig(self.directory + 'MAB_total_delay_mse', format="PNG")

    def plot_total_rewards(self, total_rewards, optimal_delay):
        # plot the total rewards
        print(f"total_rewards:{total_rewards}")
        plt.figure()
        x = range(len(total_rewards))
        print(f"total rewards array x:{x}")
        print(f"total rewards array:{total_rewards}")
        plt.plot(x, total_rewards)
        plt.xlabel("time")
        plt.ylabel("rewards of the selected optimal path")
        plt.hlines(y=optimal_delay, xmin=0, xmax=len(total_rewards), colors='red', linestyles='-', lw=2,
                   label='optimal delay')
        # plt.show()
        plt.savefig(self.directory + 'rewards', format="PNG")