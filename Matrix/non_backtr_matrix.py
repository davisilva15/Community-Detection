import numpy as np


def non_backtracking(adj_matrix):
	"""
	For a graph given by its adjacency matrix, returns its non-backtracking matrix and edge dictionary
	Time: O(|E|^2)		Space: O(|E|^2)
	"""
	# Number of nodes on the graph
	n = len(adj_matrix)

	# Dictionary giving for each (directed) edge a number from 0 to 2|E| - 1
	edge_dict = {}
	count = 0
	for r in range(1, n):
		for c in range(r + 1, n + 1):
			if adj_matrix[r - 1][c - 1]:
				edge_dict[(r, c)] = count
				edge_dict[(c, r)] = count + 1
				count += 2
	
	# Construction of the non-backtracking matrix for this graph
	non_backtr = np.zeros((count, count), dtype = np.int)
	for i in range(1, n + 1):
		for j in range(1, n + 1):
			if adj_matrix[i - 1][j - 1]:
				for k in range(1, n + 1):
					if adj_matrix[j - 1][k - 1] and i != k:
						non_backtr[edge_dict[(i, j)]][edge_dict[(j, k)]] = 1
	
	# Returns the non-backtracking matrix (numpy int matrix) and edge dictionary
	return non_backtr, edge_dict
