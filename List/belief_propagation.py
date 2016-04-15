import numpy as np
from BP_infer import BP_Inference
from BP_learn import BP_learning


def infer_assignment(file, criterium, t_max):
	"""
	Given a file containing all the graph's data, infers the group assignment by belief propagation
	Returns the (normalized) overlap to the actual group assignment and the Bethe free energy
	"""
	# File where the graph data is writen
	f = open(file, 'r')

	# All the true parameters of the graph
	N = int(f.readline())
	q = int(f.readline())
	adj_list = eval(f.readline())
	group = np.array(eval(f.readline()))
	n = np.array(eval(f.readline()))
	c = np.array(eval(f.readline()))*N

	f.close()

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


def overlap(N, q, n, groups, actual):
	"""
	Calculates the overlap between the group assignment groups and the actual group assignment of the nodes
	It's normalized to return 0 if all nodes are assigned to the same group and 1 if both assignments are identical
	"""
	# All permutations of [1, 2, ..., q]
	all_perm = __all_permutations(q)
	# The number of nodes whose group assignment is correct up to a permutation of the group labels
	max_count = 0
	for perm in all_perm:
		count = 0
		for i in range(N):
			if perm[groups[i] - 1] == actual[i]:
				count += 1
		if count > max_count:
			max_count = count
	# Proportion of correctly assigned nodes (up to a permutation of group labels)
	ovlp = max_count/N
	# Normalization of the overlap
	na = max(n)
	return (ovlp - na)/(1 - na)


def __all_permutations(N):
	"""
	Returns a list with all permutations of [1, 2, ..., N]
	"""
	if N == 1:
		return [[1]]
	permut = __all_permutations(N - 1)
	all_perm = []
	for perm in permut:
		for i in range(N):
			all_perm.append(perm[:i] + [N] + perm[i:])
	return all_perm
