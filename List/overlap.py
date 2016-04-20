import numpy as np


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
