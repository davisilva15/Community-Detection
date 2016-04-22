import numpy as np
from BP_infer import BP_Inference
from BP_learn import BP_learning
from overlap import overlap


def infer_assignment(N, q, adj_list, group, n, c, criterium, t_max):
	"""
	Given the graph's true data, infers the group assignment by belief propagation
	Returns the (normalized) overlap to the actual group assignment and the Bethe free energy
	"""
	# Initialization of the optimal values
	f_min = 0
	groups_opt = np.zeros(N, dtype = np.int8)

	# We take the best of 3 applications of BP_Inference (measured by its free energy) to wash away bad random initializations
	for _ in range(3):
		# The group assignment given by the belief propagation algorithm and its free energy
		est_prop, est_edges, groups, f_BP = BP_Inference(q, n, c, adj_list, criterium, t_max)

		# If the current free energy is smaller than the minimum free energy, we update the group assignment
		if f_BP < f_min:
			f_min = f_BP
			np.copyto(groups_opt, groups)
		
	# Returns the overlap between the infered group assignment and the actual assignment and its free energy
	return overlap(N, q, n, groups_opt, group), f_min


def learn_parameters(adj_list, q, c_max, nb_iterations, crit_infer = 0.2, crit_learn = 0.2, tmax_infer = 12, tmax_learn = 8):
	"""
	Learns the true group parameters of a given graph, and returns the optimal group assignment given by the belief
	propagation algorithm
	"""
	# Number of nodes on the graph
	N = len(adj_list)

	# Optimal values given by the algorithm
	f_min = 0
	groups_opt = np.zeros(N, dtype = np.int8)
	for _ in range(nb_iterations):
		# Random initialization of each group's size
		n = np.random.rand(q)
		n /= np.sum(n)
		
		# Random initialization of the edge matrix
		c = np.random.rand(q, q)
		for i in range(q - 1):
			for j in range(i + 1, q):
				c[j, i] = c[i, j]
		c *= c_max

		# Application of the BP_learning algorithm for these initialized values
		groups, f_BP = BP_learning(q, n, c, adj_list, crit_infer, crit_learn, tmax_infer, tmax_learn)

		# Updates the optimal values found
		if f_BP < f_min:
			f_min = f_BP
			np.copyto(groups_opt, groups)

	# Returns the optimal group assignment found
	return groups_opt
