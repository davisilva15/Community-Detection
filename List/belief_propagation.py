import numpy as np
from BP_infer import BP_Inference
from BP_learn import BP_learning
from read_list import read_file
from overlap import overlap


def infer_assignment(file, criterium, t_max):
	"""
	Given a file containing all the graph's data, infers the group assignment by belief propagation
	Returns the (normalized) overlap to the actual group assignment and the Bethe free energy
	"""
	# Reads the graph's data from file
	N, q, adj_list, group, n, c = read_file(file)

	# The true proportion of nodes in each group
	n_actual = np.copy(n)
	# Initialization of the optimal values
	f_min = 0
	groups_opt = np.zeros(N, dtype = np.int8)
	n_opt = np.zeros(q)

	# We take the best of 3 applications of BP_Inference (measured by its free energy) to wash away bad random initializations
	for _ in range(3):
		# The group assignment given by the belief propagation algorithm and its free energy
		groups, f_BP = BP_Inference(q, n, np.copy(c), adj_list, criterium, t_max)
		# If the current free energy is smaller than the minimum free energy, we update the group assignment
		if f_BP < f_min:
			f_min = f_BP
			np.copyto(groups_opt, groups)
			np.copyto(n_opt, n)
		# We return to n its initial value
		np.copyto(n, n_actual)

	# Returns the overlap between the infered group assignment and the actual assignment and its free energy
	return overlap(N, q, n_opt, groups_opt, group), f_min
