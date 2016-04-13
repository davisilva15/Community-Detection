import numpy as np
from BP_infer import BP_Inference


def BP_learning(q, n_init, c_init, adj_list, crit_infer, crit_learn, tmax_infer, tmax_learn):
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
		groups, f_BP = BP_Inference(q, n, c, adj_list, crit_infer, tmax_infer)
		
		# Updating the convergence measure
		conv = np.linalg.norm(n - n_old, 1) + np.sum(np.fabs(c - c_old))
	return groups, f_BP
