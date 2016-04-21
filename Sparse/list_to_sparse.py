import numpy as np
from scipy.sparse import csr_matrix


def list_to_sparse(adj_list):
	"""
	Given the adjacency list of a graph, returns its corresponding sparse CSR adjacency matrix
	"""
	# Number of nodes on the graph
	N = len(adj_list.keys())

	# Construction of the graph's sparse adjacency matrix
	row = []
	col = []
	for u in adj_list.keys():
		for v in adj_list[u]:
			row.append(u - 1)
			col.append(v - 1)
	
	# Returns the coordinates of the graph's adjacency matrix nonzero elements
	return row, col
