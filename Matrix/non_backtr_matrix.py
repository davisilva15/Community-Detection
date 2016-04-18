import numpy as np


def non_backtracking(adj_matrix):
	"""
	For a graph given by its adjacency matrix, returns its non-backtracking matrix and edge dictionary
	Time: O(|E|^2)		Space: O(|E|^2)
	"""
	# Number of nodes on the graph
	N = len(adj_matrix)

	# Maps linking each (directed) edge of the graph and a code from 0 to 2|E| - 1
	edge_to_code = {}
	code_to_edge = []
	count = 0
	for u in range(1, N + 1):
		for v in range(1, N + 1):
			if adj_matrix[u - 1][v - 1]:
				edge_to_code[(u, v)] = count
				code_to_edge.append((u, v))
				count += 1
	
	# Construction of the non-backtracking matrix for this graph
	non_backtr = np.zeros((count, count), dtype = np.int8)
	for u in range(1, N + 1):
		for v in range(1, N + 1):
			if adj_matrix[u - 1][v - 1]:
				for w in range(1, N + 1):
					if adj_matrix[v - 1][w - 1] and u != w:
						non_backtr[edge_to_code[(u, v)]][edge_to_code[(v, w)]] = 1
	
	# Returns the non-backtracking matrix (numpy int matrix) and edge dictionary
	return non_backtr, code_to_edge
