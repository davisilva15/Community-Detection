import numpy as np
from BP_infer import BP_Inference


def BP_learning(q, n_init, c_init, adj_list, crit_infer, crit_learn, tmax_infer, tmax_learn):
	"""
	Given a graph by its adjacency list and a number of groups to distribute its edges, along with initial assumptions of the
	group parameters, runs the belief propagation algorithm to learn the true parameters and returns the most probable group
	assignment and its free energy
	"""
	# Number of nodes on the graph
	N = len(adj_list.keys())

	# Initial values of arrays n and c
	n = n_init
	c = c_init

	conv = crit_learn + 1
	t = 0
	while conv > crit_learn and t < tmax_learn:
		t += 1

		# Values of the arrays n and c before changes are made
		n_old = np.copy(n)
		c_old = np.copy(c)

		# Most probable group of each node and the free energy of this configuration
		n, c, groups, f_BP = BP_Inference(q, n, c, adj_list, crit_infer, tmax_infer)
		
		# Updating the convergence measure
		conv = np.sum(np.fabs(n - n_old)) + np.sum(np.fabs(c - c_old))
	# Returns the estimated assignment of the nodes and its Bethe free energy
	return groups, f_BP
