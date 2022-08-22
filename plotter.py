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
        #plt.xticks(np.arange(len(Dict_edge_m)))
        plt.legend()

    def plot_edge_delay_difference_alongtime(self, s,e, edge_delay_difference_list,link_range):
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
        plt.savefig(self.directory + 'delay difference from the mean link %s' %link_range, format="PNG")

    def plot_edge_delay_difference(self,G,Dict_edge_theta):
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        langs = list(range(1,len(G.edges)+1))
        #(f"langs:{langs}")
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

    def plot_time_average_rewards(self, total_rewards, optimal_delay):
        # plot the total rewards
        # print(f"total_rewards:{total_rewards}")
        plt.figure()
        x = range(len(total_rewards))
        time_average_rewards = []
        sum = 0
        for i in range(len(total_rewards)):
            sum += total_rewards[i]
            time_average_rewards.append(sum / (i + 1))
        print(f"total rewards array x:{x}")
        print(f"total rewards array:{total_rewards}")
        plt.plot(x, time_average_rewards)
        plt.xlabel("time")
        plt.ylabel("rewards of the selected optimal path")
        plt.hlines(y=optimal_delay, xmin=0, xmax=len(total_rewards), colors='red', linestyles='-', lw=2,
                   label='optimal delay')
        # plt.show()
        plt.savefig(self.directory + 'rewards', format="PNG")

    def plot_bar_edge_exploration_training_with_increasing_monitor(self, G, monitors_list, explored_edges_num):
        #x = [str(len(monitors) /len(G.nodes)) for monitors in monitors_list]
        x = ['0.1', '0.2', '0.3', '0.4','0.5']
        y = [edges_count / len(G.edges) for edges_count in explored_edges_num]
        # print(x, y)
        fig = plt.figure(figsize=(10, 7))
        barwidth = 0.25
        plt.xlabel("% of nodes selected as monitors")
        plt.ylabel("% of explored edges")
        bar = np.array(x)
        plt.bar(bar, y, width=barwidth)
        # plt.show()
        plt.savefig(self.directory + 'MAB_edge_exploration_with_increasing_monitors.png')

    def plot_mse_with_increasing_monitor_training(self, total_edge_mse_list_with_increasing_monitors):
        labels=['0.1', '0.2', '0.3', '0.4','0.5']
        #line_num=len(total_edge_mse_list_with_increasing_monitors)
        x=range(len(total_edge_mse_list_with_increasing_monitors[0]))
        fig = plt.figure(figsize=(10, 7))
        for i in range (len(total_edge_mse_list_with_increasing_monitors)):
            plt.plot(x, total_edge_mse_list_with_increasing_monitors[i], label=labels[i])
        plt.legend()
        plt.savefig(self.directory + "mse_with_increasing_monitor_training")


    def plot_NT_verification_edge_computed_rate_with_monitors_increasing(self, G, monitors_list, solved_edges_count ):
        plt.figure()
        x = [len(monitors) / len(G.nodes) for monitors in monitors_list]
        y = [edges_count / len(G.edges) for edges_count in solved_edges_count]
        print(x, y)
        plt.plot(x, y)
        plt.xlabel("% of nodes selected as monitors")
        plt.ylabel("% of solved links")
        # plt.show()
        plt.savefig('plots/network_tomography_verification_node%s_with_link_weight=1.png'%(len(G.nodes)))

    def plot_rewards_along_with_different_monitors(self, total_rewards_list, optimal_delay):
        labels = ['0.1', '0.2', '0.3', '0.4', '0.5']
        line_num = len(total_rewards_list)
        x = range(len(total_rewards_list[0]))
        fig = plt.figure(figsize=(10, 7))
        for i in range(line_num):
            sum = 0
            y = []
            for j in range(len(total_rewards_list[i])):
                sum += total_rewards_list[i][j]
                y.append(sum / (j + 1))
            plt.plot(x, y, label=labels[i])
        plt.xlabel("time")
        plt.ylabel("time averaged rewards of the selected optimal path during training")
        plt.hlines(y=optimal_delay, xmin=0, xmax=len(total_rewards_list[0]), colors='red', linestyles='-', lw=2,
                   label='optimal delay')
        plt.legend()
        plt.savefig(self.directory + "rewards with different #minitors")

    def plot_edge_exporation_times_with_differrent_monitor_size(self, G, total_edge_exploration_during_training_list):
            edges_num = len(G.edges)
            index = range(0, edges_num)
            # index.sort()
            selected_edges_list = []
            for i in range(len(total_edge_exploration_during_training_list)):
                for j in range(len(index)):
                    print(f"index:{index[j]}")
                list = total_edge_exploration_during_training_list[i]
                selected_edges_list.append([list[index[j]] for j in range(len(index))])

            fig = plt.figure()
            ax = fig.add_subplot(111)
            # Multiple bar chart
            x = ['0.1', '0.2', '0.3', '0.4','0.5']
            index = [str(index[i]) for i in range(len(index))]
            for i in range(len(total_edge_exploration_during_training_list)):
                ax.bar(index, selected_edges_list[i], width=0.55, align='center', label=x[i])
            # Define x-ticks
            # ticks=[str(index[i]) for i in range(len(index))]
            # plt.xticks(index, ticks)
            # Layout and Display
            plt.xlabel("LinkID")
            plt.ylabel("total explored time during MAB training ")
            plt.tight_layout()
            plt.legend()
            plt.savefig(self.directory + " the exploration times of 20 random edges with different monitor numbers")

    def plot_edge_computed_during_training(self, G,average_computed_edge_during_training):
        x = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
        y = [edges_count / len(G.edges) for edges_count in average_computed_edge_during_training]
        # print(x, y)
        fig = plt.figure(figsize=(10, 7))
        barwidth = 0.25
        plt.xlabel("% of nodes selected as monitors")
        plt.ylabel("% of computed edges")
        bar = np.array(x)
        plt.bar(bar, y, width=barwidth)
        # plt.show()
        plt.savefig(self.directory + 'MAB_edge_computed_with_increasing_monitors.png')