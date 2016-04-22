import numpy as np
from graph_list import Graph
from non_backtr_list import non_backtr_cluster


def non_backtracking_test(N, q, c_avg, delta_min, delta_max, step, nb_instances):
	f = open('non_backtr_results.txt', 'w')

	f.write('Overlap obtained by the non-backtracking spectral algorithm on graphs with {} groups and average degree {}.'.format(q, c_avg))
	f.write('\nEvery group has the same number of nodes, and each result shown is for an independent random SBM graph with {} nodes.'.format(N))

	nb_vector = np.array([N//q for _ in range(q)])
	for delta in np.arange(delta_min, delta_max, step):
		sum_ovlp = 0
		edge_matrix = construct_edge_matrix(q, c_avg, delta)
		f.write('\n\nResults for c_in = {:.3f}, c_out = {:.3f}:'.format(edge_matrix[0][0], edge_matrix[0][1]))
		for i in range(1, nb_instances + 1):
			g = Graph(np.copy(nb_vector), np.copy(edge_matrix)/N)
			ovlp = non_backtr_cluster(g.nb_nodes, g.nb_groups, g.adj_list, g.group, g.group_prop)
			sum_ovlp += ovlp
			f.write('\n\tOverlap from test {}:'.format(i))
			f.write(' {:.3f}'.format(ovlp))
		f.write('\n\t\tAverage overlap: {:.3f}'.format(sum_ovlp/nb_instances))

	f.close()


def construct_edge_matrix(q, c_avg, delta):
	c_out = c_avg - delta/q
	edge_matrix = c_out*np.ones((q, q)) + delta*np.eye(q)
	return edge_matrix
