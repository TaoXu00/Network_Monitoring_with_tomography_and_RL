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
        plt.close()
    def plot_edge_exploitation_times_bar(self,label,Dict_edge_m):
        fig = plt.figure(figsize=(10, 7))
        plt.xlabel('linkID', fontweight='bold', fontsize=15)
        plt.ylabel('#times the edge is observed', fontweight='bold', fontsize=15)
        # Horizontal Bar Plot
        bar=np.arange(len(Dict_edge_m))
        plt.bar(bar,Dict_edge_m.values(), label= label)
        #plt.xticks(np.arange(len(Dict_edge_m)))
        plt.legend()
        plt.close()


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
        plt.close()

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
        plt.close()


    def plot_total_edge_delay_mse(self,total_mse_array):
        # plot the mse
        plt.figure()
        x = range(len(total_mse_array))
        #print(f"mse:{x}")
        #print(f"mse:{total_mse_array}")
        plt.plot(x, total_mse_array)
        plt.xlabel("time slot")
        plt.ylabel("total_mse of all edges dalay")
        # plt.show()
        plt.savefig(self.directory + 'MAB_total_delay_mse', format="PNG")
        plt.close()

    def plot_total_optimal_edge_delay_mse(self, total_mse_optimal_edges_array):
        plt.figure()
        x = range(len(total_mse_optimal_edges_array))
        # print(f"mse:{x}")
        # print(f"mse:{total_mse_array}")
        plt.plot(x, total_mse_optimal_edges_array)
        plt.xlabel("time")
        plt.ylabel("learning error of links in optimal paths")
        # plt.show()
        plt.savefig(self.directory + 'MAB_Learning_Error_of_Links_in_Optimal_Paths.png', format="PNG")
        plt.close()

    def plot_time_average_rewards(self, mse_list):
        # plot the total rewards
        # print(f"total_rewards:{total_rewards}")
        plt.figure()
        x = range(len(mse_list))
        plt.plot(x, mse_list)
        plt.xlabel("time")
        plt.ylabel("mse of rewards of selected optimal path among monitors")
        # plt.show()
        plt.savefig(self.directory + 'mse rewards', format="PNG")
        plt.close()

    def plot_bar_edge_exploration_training_with_increasing_monitor(self, monitors_deployment_percentage, explored_edges_rate):
        #x = [str(len(monitors) /len(G.nodes)) for monitors in monitors_list]
        x=['0.1', '0.2', '0.3', '0.4','0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
        y = explored_edges_rate
        # print(x, y)
        fig = plt.figure(figsize=(10, 7))
        barwidth = 0.25
        plt.xlabel("% of nodes selected as monitors")
        plt.ylabel("% of explored edges")
        bar = np.array(x)
        plt.bar(bar, y, width=barwidth)
        # plt.show()
        plt.savefig(self.directory + 'MAB_edge_exploration_with_increasing_monitors.png')
        plt.close()

    def plot_total_edge_delay_mse_with_increasing_monitor_training(self, monitors_deployment_percentage, total_edge_mse_list_with_increasing_monitors):
        labels = []
        for per in monitors_deployment_percentage:
            labels.append(str(per) + '%')
        #line_num=len(total_edge_mse_list_with_increasing_monitors)
        x = range(len(total_edge_mse_list_with_increasing_monitors[0]))
        fig = plt.figure()
        plt.rcParams.update({'font.size': 13})

        colors = ['firebrick', 'cornflowerblue', 'goldenrod', 'forestgreen', 'darkmagenta']
        linestyles = ['dotted', 'dashed', 'dashdot', (0, (3, 5, 1, 5, 1, 5)), 'solid']
        for i in range(len(total_edge_mse_list_with_increasing_monitors)):
            plt.plot(x, total_edge_mse_list_with_increasing_monitors[i], label=labels[i], color=colors[i],
                     linestyle=linestyles[i])
        plt.xlabel("learning time")
        plt.ylabel("MSE of link delay during learning")
        plt.legend(fontsize=13)
        # plt.grid(True)
        plt.savefig(self.directory + "MSE_of_total_links_delay_with_increasing_monitor_training")
        plt.close()

    def plot_total_optimal_edge_delay_mse_with_increasing_monitor_training(self, monitors_deployment_percentage, multi_times_avg_mse_total_optimal_links_delay_array):
        labels = []
        for per in monitors_deployment_percentage:
            labels.append(str(per) + '%')
        # line_num=len(total_edge_mse_list_with_increasing_monitors)
        total_edge_mse_list_with_increasing_monitors = multi_times_avg_mse_total_optimal_links_delay_array
        x = range(len(total_edge_mse_list_with_increasing_monitors[0]))
        # x = range(len(total_edge_mse_list_with_increasing_monitors))
        fig = plt.figure()
        plt.rcParams.update({'font.size': 13})

        colors = ['cornflowerblue','goldenrod','forestgreen', 'firebrick','purple']
        #linestyles = ['dotted', 'dashed', 'dashdot', (0, (3, 5, 1, 5, 1, 5)), 'solid']
        markers=["s", "^", "+", "p", "x" ]
        for i in range(len(total_edge_mse_list_with_increasing_monitors)):
            plt.plot(x, total_edge_mse_list_with_increasing_monitors[i], label=labels[i], color=colors[i],
                     marker=markers[i])
        plt.xlabel("learning time")
        plt.ylabel("MSE of link delay during learning")
        plt.legend(fontsize=13)
        # plt.grid(True)
        plt.savefig(self.directory + "MSE_of_total_optimal_links_delay_with_increasing_monitor_training.png")
        plt.close()


    def plot_total_edge_delay_mse_with_increasing_monitor_training_from_file(self, monitors_deployment_percentage, filename):
        labels = []
        for per in monitors_deployment_percentage:
            labels.append(str(per) + '%')
        #line_num=len(total_edge_mse_list_with_increasing_monitors)
        total_edge_mse_list_with_increasing_monitors=np.loadtxt(filename, dtype=float)
        x=range(len(total_edge_mse_list_with_increasing_monitors[0]))
        #x = range(len(total_edge_mse_list_with_increasing_monitors))
        fig = plt.figure()
        plt.rcParams.update({'font.size': 13})

        colors=['firebrick', 'cornflowerblue', 'goldenrod', 'forestgreen', 'darkmagenta']
        linestyles = ['dotted','dashed', 'dashdot', (0, (3, 5, 1, 5, 1, 5)),'solid']
        for i in range (len(total_edge_mse_list_with_increasing_monitors)):
            plt.plot(x, total_edge_mse_list_with_increasing_monitors[i], label=labels[i], color=colors[i], linestyle= linestyles[i])
        plt.xlabel("learning time")
        plt.ylabel("MSE of link delay during learning")
        plt.legend(fontsize=13)
        #plt.grid(True)
        plt.savefig(self.directory + "MSE_of_total_links_delay_with_increasing_monitor_training")
        plt.close()

    def plot_avg_traffic_overhead_every_200_iterations(self, monitors_deployment_percentage, multi_times_avg_traffic_overhead_every_200_iterations_with_increasing_monitors):
        labels = []
        for per in monitors_deployment_percentage:
            labels.append(str(per) + '%')
        print("new_avg_with_increasing_monitors row num: %d:" % (
            len(multi_times_avg_traffic_overhead_every_200_iterations_with_increasing_monitors)))
        print("new_avg_with_increasing_monitors column num %d:" % (
            len(multi_times_avg_traffic_overhead_every_200_iterations_with_increasing_monitors)))

        x = np.arange(200, 3200, 200)
        fig = plt.figure()
        plt.rcParams.update({'font.size': 13})

        colors = ['cornflowerblue', 'goldenrod', 'forestgreen', 'firebrick', 'purple']
        # linestyles = ['dotted', 'dashed', 'dashdot', (0, (3, 5, 1, 5, 1, 5)), 'solid']
        markers = ["s", "^", "+", "p", "x"]
        for i in range(len(multi_times_avg_traffic_overhead_every_200_iterations_with_increasing_monitors)):
            plt.plot(x, multi_times_avg_traffic_overhead_every_200_iterations_with_increasing_monitors[i],
                     label=labels[i], color=colors[i],
                     marker=markers[i])
        plt.xlabel("time")
        plt.ylabel(" Traffic Overhead")
        plt.legend(fontsize=13)
        plt.grid(True)
        plt.savefig(self.directory + "Traffic Overhead.png")
        plt.close()

    def plot_avg_path_oscilation_every_200_times(self, monitors_deployment_percentage,multi_times_avg_path_oscilation_array):
        labels = []
        for per in monitors_deployment_percentage:
            labels.append(str(per) + '%')
        print("new_avg_with_increasing_monitors row num: %d:" % (len(multi_times_avg_path_oscilation_array)))
        print("new_avg_with_increasing_monitors column num %d:" % (len(multi_times_avg_path_oscilation_array[0])))

        x = np.arange(200, 3200, 200)
        fig = plt.figure()
        plt.rcParams.update({'font.size': 13})

        colors = ['cornflowerblue', 'goldenrod', 'forestgreen', 'firebrick', 'purple']
        # linestyles = ['dotted', 'dashed', 'dashdot', (0, (3, 5, 1, 5, 1, 5)), 'solid']
        markers = ["s", "^", "+", "p", "x"]
        for i in range(len(multi_times_avg_path_oscilation_array)):
            plt.plot(x, multi_times_avg_path_oscilation_array[i], label=labels[i], color=colors[i],
                     marker=markers[i])
        plt.xlabel("learning time")
        plt.ylabel(" # of Path Oscillation")
        plt.legend(fontsize=13)
        # plt.grid(True)
        plt.savefig(self.directory + "# of Path Oscilation.png")
        plt.close()

    def plot_avg_optimal_actions_every_200_times(self,monitors_deployment_percentage,multi_avg_optimal_actions_with_increasing_monitors):
        labels = []
        for per in monitors_deployment_percentage:
            labels.append(str(per) + '%')
        new_avg_with_increasing_monitors=[]
        for rate_array in multi_avg_optimal_actions_with_increasing_monitors:
            avg_100=np.average(np.array(rate_array).reshape(-1,200),axis=1)
            new_avg_with_increasing_monitors.append(avg_100)
        print("new_avg_with_increasing_monitors row num: %d:" %(len(new_avg_with_increasing_monitors)))
        print("new_avg_with_increasing_monitors column num %d:" %(len(new_avg_with_increasing_monitors[0])))

        x=np.arange(200,3200,200)
        fig = plt.figure()
        plt.rcParams.update({'font.size': 13})

        colors = ['cornflowerblue', 'goldenrod', 'forestgreen', 'firebrick', 'purple']
        # linestyles = ['dotted', 'dashed', 'dashdot', (0, (3, 5, 1, 5, 1, 5)), 'solid']
        markers = ["s", "^", "+", "p", "x"]
        for i in range(len(new_avg_with_increasing_monitors)):
            plt.plot(x, new_avg_with_increasing_monitors[i], label=labels[i], color=colors[i],
                     marker=markers[i])
        plt.xlabel("learning time")
        plt.ylabel("Rate of optimal actions (%)")
        plt.legend(fontsize=13)
        # plt.grid(True)
        plt.savefig(self.directory + "Rate_of_optimal_actions_with_increasing_monitor_training.png")
        plt.close()



    def plot_NT_verification_edge_computed_rate_with_monitors_increasing(self, G, monitors_list, solved_edges_count ):
        plt.figure()
        x = [len(monitors) / len(G.nodes) for monitors in monitors_list]
        y = [edges_count / len(G.edges) for edges_count in solved_edges_count]
        #print(x, y)
        plt.plot(x, y)
        plt.xlabel("% of nodes selected as monitors")
        plt.ylabel("% of solved links")
        # plt.show()
        plt.savefig('plots/network_tomography_verification_node%s_with_link_weight=1.png'%(len(G.nodes)))
        plt.close()

    def plot_rewards_mse_along_with_different_monitors(self,monitors_deployment_percentage,total_rewards_mse_list):
        labels=[]
        for per in monitors_deployment_percentage:
            labels.append(str(per)+'%')
        line_num = len(total_rewards_mse_list)
        x = range(len(total_rewards_mse_list[0]))
        fig = plt.figure(figsize=(10, 7))
        for i in range(line_num):
            plt.plot(x, total_rewards_mse_list[i], label=labels[i])
        plt.xlabel("time")
        plt.ylabel("mse of time averaged rewards of the selected optimal paths during training")
        plt.legend()
        plt.savefig(self.directory + "rewards mse with different #minitors")
        plt.close()

    def plot_edge_exporation_times_with_differrent_monitor_size(self, G, total_edge_exploration_during_training_list):
            edges_num = len(G.edges)
            index = range(0, edges_num)
            # index.sort()
            selected_edges_list = []
            for i in range(len(total_edge_exploration_during_training_list)):
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
            plt.close()

    def plot_edge_computed_rate_during_training(self,monitors_deployment_percentage,average_computed_edge_rate_during_training):
        x = []
        for per in monitors_deployment_percentage:
            x.append(str(per))
        y = average_computed_edge_rate_during_training
        fig = plt.figure(figsize=(10, 7))
        barwidth = 0.25
        plt.xlabel("% of nodes selected as monitors")
        plt.ylabel("% of computed edges")
        bar = np.array(x)
        plt.bar(bar, y, width=barwidth)
        plt.savefig(self.directory + 'MAB_edge_computed_rate_with_increasing_monitors.png')
        plt.close()

    def plot_edge_computed_rate_with_different_topology_size(self):
        percentage = ['0.1', '0.2', '0.3', '0.4', '0.5','0.6','0.7','0.8','0.9','1.0']
        edge_comput_rate_20nodes=[0.0, 0.05796490043874452, 0.10419244759440507, 0.11742978212772341, 0.2896838789515131, 0.3347301908726141, 0.5024843439457007, 0.7434657066786665,0.7759215509806128, 0.8975794052574343]
        edge_compute_number_50nodes=[1.256631071305546,6.163623837409577,20.436445056837755,30.779193937306236,43.56286600068894,64.29968997588702,64.19910437478471,73.87599035480537,81.72959007922839, 79.74061315880124]
        edge_compute_rate_50nodes=[x/96 for x in edge_compute_number_50nodes]
        barWidth = 0.25
        fig = plt.subplots()
        # set height of bar
        nodes_20 = edge_comput_rate_20nodes
        nodes_50 = edge_compute_rate_50nodes
        # Set position of bar on X axis
        br1 = np.arange(len(percentage))
        br2 = [x + barWidth for x in br1]

        # Make the plot
        plt.bar(br1, nodes_20, color='r', width=barWidth,
                edgecolor='grey', label='20 nodes')
        plt.bar(br2, nodes_50, color='g', width=barWidth,
                edgecolor='grey', label='50 nodes')


        # Adding Xticks
        plt.xlabel('%of nodes selected as monitors',  fontsize=15)
        plt.ylabel('%of the identified links',  fontsize=15)
        plt.xticks(np.arange(len(percentage)),percentage)
        plt.legend()
        plt.savefig(self.directory + '%of identified edges with different topology size and different number of monitors' , format="PNG")
        plt.close()

    def plot_average_regrets(self, averaged_regret_list):
        plt.figure()
        x = range(len(averaged_regret_list))
        plt.plot(x, averaged_regret_list)
        plt.xlabel("time")
        plt.ylabel("averaged regret of selected shortest path among monitors")
        plt.savefig(self.directory + 'averaged regret', format="PNG")
        plt.close()

    def plot_rate_of_correct_shortest_path(self, correct_shortest_path_selected_rate):
        plt.figure()
        x = range(len(correct_shortest_path_selected_rate))
        plt.plot(x, correct_shortest_path_selected_rate)
        plt.xlabel("time")
        plt.ylabel("rate of correctly selected shortest path among monitors")
        plt.savefig(self.directory + 'rate of correctly selected shortest path among monitors', format="PNG")
        plt.close()

    def plot_edge_delay_difference_for_some_edges(self, optimal_edges_delay_difference_after_inti,
                                                  optimal_edges_delay_difference_after_training):
        barWidth = 0.25
        fig = plt.subplots()
        # set height of bar
        init = optimal_edges_delay_difference_after_inti
        after_training = optimal_edges_delay_difference_after_training
        br1 = np.arange(len(init))
        br2 = [x + barWidth for x in br1]
        # Make the plot
        plt.bar(br1, init, color='r', width=barWidth,
                edgecolor='grey', label='after_init')
        plt.bar(br2, after_training, color='g', width=barWidth,
                edgecolor='grey', label='after_training')

        # Adding Xticks
        xlable = np.arange(len(init))
        plt.xlabel('linkID', fontweight='bold', fontsize=15)
        plt.ylabel('delay difference from the mean', fontweight='bold', fontsize=15)
        plt.xticks(xlable)
        plt.legend()
        plt.savefig(self.directory + 'delay difference of optimal edges from mean after init and after training')
        plt.close()

    def plot_diff_from_optimal_path_of_selected_shortest_paths(self, abs_diff_of_delay_from_optimal):
        plt.figure()
        x = range(len(abs_diff_of_delay_from_optimal))
        plt.plot(x, abs_diff_of_delay_from_optimal)
        plt.xlabel("time")
        plt.ylabel("mse of the selected shortest path from optimal shortest path")
        plt.savefig(self.directory + 'absolute difference of the selected shortest path from optimal shortest path', format="PNG")
        plt.close()

    def plot_optimal_path_selected_percentage_list_with_increasing_monitors(self, monitors_deployment_percentage, optimal_path_selected_rate):
        x=monitors_deployment_percentage
        x_label = [str(pert) for pert in monitors_deployment_percentage]
        y = optimal_path_selected_rate
        fig = plt.figure(figsize=(10, 7))
        barwidth = 0.25
        plt.xlabel("% of nodes selected as monitors")
        plt.ylabel(" % of the optimal paths selected")
        bar = np.arange(len(x_label))
        plt.bar(bar, y, width=barwidth)
        plt.xticks(bar,x_label)
        plt.savefig(self.directory + 'MAB_edge_computed_rate_with_increasing_monitors.png')
        plt.close()

    def plot_abs_diff_path_delay_from_the_optimal(self, monitors_deployment_percentage, optimal_path_selected_percentage_list):
        x=monitors_deployment_percentage
        x_label = [str(pert) for pert in monitors_deployment_percentage]
        y = optimal_path_selected_percentage_list
        fig = plt.figure(figsize=(10, 7))
        barwidth = 0.25
        plt.xlabel("% of nodes selected as monitors")
        plt.ylabel(" abs error from the optimal shortest paths")
        bar = np.arange(len(x_label))
        plt.bar(bar, y, width=barwidth)
        plt.xticks(bar, x_label)
        plt.savefig(self.directory + 'abs error from the optimal shortest paths.png')
        plt.legend()
        plt.close()
    def plot_percentage_of_optimal_path_selected_rate_BR_50nodes_line(self, monitors_deployment_percentage, subito_op_rate, UCB1_op_rate, subito_perfect_op_rate,BoundNT_op_rate):
        x = np.arange(10, 60, 10)
        fig = plt.figure()
        plt.rcParams.update({'font.size': 13})
        plt.grid(True)
        colors = ['cornflowerblue', 'goldenrod', 'forestgreen', 'firebrick', 'purple']
        # linestyles = ['dotted', 'dashed', 'dashdot', (0, (3, 5, 1, 5, 1, 5)), 'solid']
        markers = ["s", "^", "+", "p", "x"]
        plt.plot(x, subito_perfect_op_rate, label='Subito*', color=colors[0],
                     marker=markers[0])
        plt.plot(x, subito_op_rate, label='Subito', color=colors[1],
                 marker=markers[1])
        plt.plot(x, UCB1_op_rate, label='UCB1', color=colors[2],
                 marker=markers[2])
        plt.plot(x, BoundNT_op_rate, label='BoundNT', color=colors[3],
                 marker=markers[3])
        plt.xticks(x)
        plt.xlabel("% of nodes selected as monitors")
        plt.ylabel("Rate of optimal actions (%)")
        plt.legend(fontsize=13)
        # plt.grid(True)
        plt.savefig(self.directory + "Rate_of_optimal_actions_with_increasing_monitor_training_new.png")
        plt.close()

    def plot_abs_delay_of_optimal_path_selected_from_mean_BR_50nodes_line(self, monitors_deployment_percentage, subito_diff, UCB1_diff, subito_perfect_diff, BoundNT_diff):
        x = np.arange(10, 60, 10)
        fig = plt.figure()
        plt.grid(True)
        plt.rcParams.update({'font.size': 13})
        colors = ['cornflowerblue', 'goldenrod', 'forestgreen', 'firebrick', 'purple']
        # linestyles = ['dotted', 'dashed', 'dashdot', (0, (3, 5, 1, 5, 1, 5)), 'solid']
        markers = ["s", "^", "+", "p", "x"]
        plt.plot(x, subito_perfect_diff, label='Subito*', color=colors[0],
                 marker=markers[0])
        plt.plot(x, subito_diff, label='Subito', color=colors[1],
                 marker=markers[1])
        plt.plot(x, UCB1_diff, label='UCB1', color=colors[2],
                 marker=markers[2])
        plt.plot(x, BoundNT_diff, label='BoundNT', color=colors[3],
                 marker=markers[3])
        plt.xticks(x)
        plt.xlabel("% of nodes selected as monitors")
        plt.ylabel("Avg. regret (msec)")
        plt.legend(fontsize=13)
        plt.savefig(self.directory + "Regret_of_optimal_actions_with_increasing_monitor_training_new.png")
        plt.close()

    def plot_percentage_of_optimal_path_selected_rate_BR_50nodes(self,monitors_deployment_percentage, subito_op_rate, UCB1_op_rate, subito_perfect_op_rate):
        barWidth = 0.25
        fig = plt.figure(figsize=(10, 10))

        # set height of bar
        x=monitors_deployment_percentage
        x_label = [str(pert) for pert in monitors_deployment_percentage]
        br1 = np.arange(len(UCB1_op_rate))
        br2 = [x + barWidth for x in br1]
        br3= [x + barWidth for x in br2]
        for i in range(len(UCB1_op_rate)):
            UCB1_op_rate[i]=UCB1_op_rate[i]*100
            subito_op_rate[i]=subito_op_rate[i]*100
            subito_perfect_op_rate[i]=subito_perfect_op_rate[i]*100
        # Make the plot
        plt.rcParams.update({'font.size': 30})
        plt.bar(br1, UCB1_op_rate,  width=barWidth,
                edgecolor='grey', label='UCB1', hatch='/')
        plt.bar(br2,subito_op_rate,  width=barWidth,
                edgecolor='grey', label='Subito', hatch='o')
        plt.bar(br3, subito_perfect_op_rate, width=barWidth,
                edgecolor='grey', label='Subito*', hatch='*')

        # Adding Xticks
        plt.xlabel('% of nodes selected as monitors')
        plt.ylabel('Freq. of optimal action (%)')
        plt.xticks(br1, x_label)
        plt.legend(fontsize=25, loc='lower left')
        plt.savefig(self.directory + 'Scability_of Minitor_op_rate')
        plt.close()

    def plot_abs_delay_of_optimal_path_selected_from_mean_BR_50nodes(self,monitors_deployment_percentage,subito_diff, UCB1_diff, subito_perfect_diff):
        barWidth = 0.25
        fig = plt.figure(figsize=(10, 10))
        # set height of bar
        x = monitors_deployment_percentage
        x_label = [str(pert) for pert in monitors_deployment_percentage]
        br1 = np.arange(len(UCB1_diff))
        br2 = [x + barWidth for x in br1]
        br3 = [x + barWidth for x in br2]
        # Make the plot
        plt.rcParams.update({'font.size': 30})
        plt.bar(br1, UCB1_diff,  width=barWidth,
                edgecolor='grey', label='UCB1', hatch='/')
        plt.bar(br2, subito_diff,  width=barWidth,
                edgecolor='grey', label='Subito', hatch='o')
        plt.bar(br3, subito_perfect_diff,  width=barWidth,
                edgecolor='grey', label='Subito*', hatch='*')

        # Adding Xticks
        plt.xlabel('% of nodes selected as monitors')
        plt.ylabel('Avg. regret (msec)')
        plt.xticks(br1, x_label)
        plt.legend(fontsize=25, loc='lower left')
        plt.savefig(self.directory + "Scability_of Minitor_delay_diff")
        plt.close()

    def plot_percentage_of_optimal_path_selected_rate_for_various_network_size_line(self, topology_size, subito_op_rate, UCB1_op_rate, subito_perfect_op_rate, boundNT_op_rate):
        x= topology_size
        fig = plt.figure()
        plt.rcParams.update({'font.size': 13})
        plt.grid(True)
        colors = ['cornflowerblue', 'goldenrod', 'forestgreen', 'firebrick', 'purple']
        # linestyles = ['dotted', 'dashed', 'dashdot', (0, (3, 5, 1, 5, 1, 5)), 'solid']
        markers = ["s", "^", "+", "p", "x"]
        plt.plot(x, subito_perfect_op_rate, label='Subito*', color=colors[0],
                 marker=markers[0])
        plt.plot(x, subito_op_rate, label='Subito', color=colors[1],
                 marker=markers[1])
        plt.plot(x, UCB1_op_rate, label='UCB1', color=colors[2],
                 marker=markers[2])
        plt.plot(x, boundNT_op_rate, label='BoundNT', color=colors[3],
                 marker=markers[3])
        plt.xticks(x)
        plt.xlabel("network size")
        plt.ylabel("Freq. of optimal actions (%)")
        plt.legend(fontsize=13)
        plt.grid(True)
        plt.savefig(self.directory + "Rate_of_optimal_actions_with_various_network_size_new.png")
        plt.close()
    def plot_abs_delay_of_optimal_path_selected_for_various_network_size_line(self, topology_size, subito_diff, UCB1_diff,subito_perfect_diff, boundNT_diff):
        x = topology_size
        fig = plt.figure()
        plt.rcParams.update({'font.size': 13})
        plt.grid(True)
        colors = ['cornflowerblue', 'goldenrod', 'forestgreen', 'firebrick', 'purple']
        # linestyles = ['dotted', 'dashed', 'dashdot', (0, (3, 5, 1, 5, 1, 5)), 'solid']
        markers = ["s", "^", "+", "p", "x"]
        plt.plot(x, subito_perfect_diff, label='Subito*', color=colors[0],
                 marker=markers[0])
        plt.plot(x, subito_diff, label='Subito', color=colors[1],
                 marker=markers[1])
        plt.plot(x, UCB1_diff, label='UCB1', color=colors[2],
                 marker=markers[2])
        plt.plot(x, boundNT_diff, label='BoundNT', color=colors[3],
                 marker=markers[3])
        plt.xticks(x)
        plt.xlabel("network size")
        plt.ylabel("Avg. regret (msec)")
        plt.legend(fontsize=13)
        plt.grid(True)
        plt.savefig(self.directory + "Abs_diff_from_optimal_actions_with_various_network_size_new.png")
        plt.close()
    def plot_percentage_of_optimal_path_selected_rate_for_varius_network_size(self, topology_size, subito_op_rate, UCB1_op_rate, subito_perfect_op_rate):
        barWidth = 0.25
        fig = plt.figure(figsize=(10, 10))
        # set height of bar
        x_label = [str(size) for size in topology_size]
        br1 = np.arange(len(UCB1_op_rate))
        br2 = [x + barWidth for x in br1]
        br3 = [x + barWidth for x in br2]
        for i in range(len(UCB1_op_rate)):
            UCB1_op_rate[i] = UCB1_op_rate[i] * 100
            subito_op_rate[i] = subito_op_rate[i] * 100
            subito_perfect_op_rate[i] = subito_perfect_op_rate[i] * 100
        # Make the plot
        plt.rcParams.update({'font.size': 30})
        plt.bar(br1, UCB1_op_rate, width=barWidth,
                edgecolor='grey', label='UCB1', hatch='/')
        plt.bar(br2, subito_op_rate, width=barWidth,
                edgecolor='grey', label='Subito', hatch='o')
        plt.bar(br3, subito_perfect_op_rate, width=barWidth,
                edgecolor='grey', label='Subito*', hatch='*')

        # Adding Xticks
        plt.xlabel('network size')
        plt.ylabel('Freq. of optimal action (%)')
        plt.xticks(br1, x_label)
        plt.legend(fontsize=25, loc='lower left')
        plt.savefig(self.directory + 'scalability_of_network_size_op_rate')
        plt.close()

    def plot_abs_delay_of_optimal_path_selected_for_various_network_size(self,topology_size, subito_diff, UCB1_diff, subito_perfect_diff):
        barWidth = 0.25
        fig = plt.figure(figsize=(10, 10))
        # set height of bar
        x_label = [str(size) for size in topology_size]
        br1 = np.arange(len(UCB1_diff))
        br2 = [x + barWidth for x in br1]
        br3 = [x + barWidth for x in br2]
        # Make the plot
        plt.rcParams.update({'font.size': 30})
        plt.bar(br1, UCB1_diff, width=barWidth,
                edgecolor='grey', label='UCB1', hatch='/')
        plt.bar(br2, subito_diff, width=barWidth,
                edgecolor='grey', label='Subito', hatch='o')
        plt.bar(br3, subito_perfect_diff, width=barWidth,
                edgecolor='grey', label='Subito*', hatch='*')

        # Adding Xticks
        plt.xlabel('network size')
        plt.ylabel('Avg. regret (msec)')
        plt.xticks(br1, x_label)
        plt.legend(fontsize=25, loc='upper left')
        plt.savefig(self.directory + 'Scability_of_network_size_delay_diff')
        plt.close()

    def plot_traffic_overhead_BR_50nodes(self,monitors_deployment_percentage, subito_NT_traffic_overhead, subito_MAB_trffic_overhead, UCB1_traffic_overhead):
        barWidth = 0.25
        fig = plt.figure(figsize=(10, 10))
        # set height of bar
        x = monitors_deployment_percentage
        x_label = [str(pert) for pert in monitors_deployment_percentage]
        br1 = np.arange(len(UCB1_traffic_overhead))
        br2 = [x + barWidth for x in br1]
        br3 = [x + barWidth for x in br2]
        # Make the plot
        plt.rcParams.update({'font.size': 30})
        plt.bar(br1, UCB1_traffic_overhead,  width=barWidth,
                edgecolor='grey', label='UCB1', hatch='/')
        plt.bar(br2, subito_MAB_trffic_overhead,  width=barWidth,
                edgecolor='grey', label='Subito_MAB', hatch='o')
        plt.bar(br3, subito_NT_traffic_overhead, width=barWidth,
                edgecolor='grey', label='Subito_NT', hatch='*')

        # Adding Xticks
        plt.xlabel('% of nodes selected as monitors')
        plt.ylabel('Traffic overhead')
        plt.xticks(br1, x_label)
        plt.legend(fontsize=25, loc='lower left')
        plt.savefig(self.directory + "Traffic overhead for BR50 with increasing monitor size")
        plt.close()

    def plot_traffic_overhead_for_various_network_size(self, topology_size, subito_MAB_trffic_overhead, subito_NT_traffic_overhead, UCB1_traffic_overhead):
        barWidth = 0.25
        fig = plt.figure(figsize=(10, 10))
        # set height of bar
        x_label = [str(size) for size in topology_size]
        br1 = np.arange(len(subito_MAB_trffic_overhead))
        br2 = [x + barWidth for x in br1]
        br3 = [x + barWidth for x in br2]
        # Make the plot
        plt.rcParams.update({'font.size': 30})
        plt.bar(br1, UCB1_traffic_overhead, width=barWidth,
                edgecolor='grey', label='UCB1', hatch='/')
        plt.bar(br2, subito_MAB_trffic_overhead, width=barWidth,
                edgecolor='grey', label='Subito_MAB', hatch='o')
        plt.bar(br3, subito_NT_traffic_overhead, width=barWidth,
                edgecolor='grey', label='Subito_NT', hatch='*')

        # Adding Xticks
        plt.xlabel('network size')
        plt.ylabel('Traffic overhead')
        plt.xticks(br1, x_label)
        plt.legend(fontsize=25, loc='upper left')
        plt.savefig(self.directory + 'Scability_of_network_size_traffic_overhead')
        plt.close()

    def plot_percentage_of_optimal_path_selected_rate_BTN(self, monitors_deployment_percentage, subito_op_rate, UCB1_op_rate, subito_perfect_op_rate ):
        barWidth = 0.25
        fig = plt.figure(figsize=(10, 10))
        # set height of bar
        x = monitors_deployment_percentage
        x_label = [str(pert) for pert in monitors_deployment_percentage]
        br1 = np.arange(len(UCB1_op_rate))
        br2 = [x + barWidth for x in br1]
        br3 = [x + barWidth for x in br2]
        for i in range(len(UCB1_op_rate)):
            UCB1_op_rate[i] = UCB1_op_rate[i] * 100
            subito_op_rate[i] = subito_op_rate[i] * 100
            subito_perfect_op_rate[i] = subito_perfect_op_rate[i] * 100
        # Make the plot
        plt.rcParams.update({'font.size': 30})
        plt.bar(br1, UCB1_op_rate, width=barWidth,
                edgecolor='grey', label='UCB1', hatch='/')
        plt.bar(br2, subito_op_rate, width=barWidth,
                edgecolor='grey', label='Subito', hatch='o')
        plt.bar(br3, subito_perfect_op_rate, width=barWidth,
                edgecolor='grey', label='Subito*', hatch='*')

        # Adding Xticks
        plt.xlabel('% of nodes selected as monitors')
        plt.ylabel('Freq. of optimal action (%)')
        plt.xticks(br1, x_label)
        plt.legend(fontsize=25, loc='lower left')
        plt.savefig(self.directory + 'Scability_of Minitor_op_rate_BTN')
        plt.close()

    def plot_abs_delay_of_optimal_path_selected_from_mean_BTN(self,monitors_deployment_percentage,subito_diff,UCB1_diff, subito_perfect_diff):
        barWidth = 0.25
        fig = plt.figure(figsize=(10, 10))
        # set height of bar
        x = monitors_deployment_percentage
        x_label = [str(pert) for pert in monitors_deployment_percentage]
        br1 = np.arange(len(UCB1_diff))
        br2 = [x + barWidth for x in br1]
        br3 = [x + barWidth for x in br2]
        # Make the plot
        plt.rcParams.update({'font.size': 30})
        plt.bar(br1, UCB1_diff, width=barWidth,
                edgecolor='grey', label='UCB1', hatch='/')
        plt.bar(br2, subito_diff, width=barWidth,
                edgecolor='grey', label='Subito', hatch='o')
        plt.bar(br3, subito_perfect_diff, width=barWidth,
                edgecolor='grey', label='Subito*', hatch='*')

        # Adding Xticks
        plt.xlabel('% of nodes selected as monitors')
        plt.ylabel('Avg. regret (msec)')
        plt.xticks(br1, x_label)
        plt.legend(fontsize=25, loc='lower left')
        plt.savefig(self.directory + "Scability_of Minitor_delay_diff_BTN")
        plt.close()
