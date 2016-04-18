import numpy as np


def list_to_matrix(adj_list):
	"""
	Given the adjacency list of a graph, returns its corresponding adjacency matrix
	"""
	# Number of nodes on the graph
	N = len(adj_list.keys())
	# The adjacency matrix has type np.array(np.int8)
	adj_matrix = np.zeros((N, N), dtype = np.int8)

	# For every edge (u, v), the matrix has 1 on its entries [u - 1][v - 1] and [v - 1][u - 1]
	for u in adj_list.keys():
		for v in adj_list[u]:
			adj_matrix[u - 1][v - 1] = 1
	
	# Returns the graph's adjacency matrix
	return adj_matrix
