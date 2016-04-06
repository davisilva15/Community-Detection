import numpy as np


def non_backtracking(adj_list):
	"""
	For a graph given by its adjacency list, returns its non-backtracking matrix and edge dictionary
	Time: O(|E|^2)		Space: O(|E|^2)
	"""
	# Number of nodes on the graph
	n = len(adj_list.keys())

	# Dictionary giving for each (directed) edge a number from 0 to 2|E| - 1
	edge_dict = {}
	count = 0
	for u in adj_list.keys():
		for v in adj_list[u]:
			edge_dict[(u, v)] = count
			count += 1

	# Construction of the non-backtracking matrix for this graph
	non_backtr = np.zeros((count, count), dtype = np.int)
	for u in adj_list.keys():
		for v in adj_list[u]:
			for w in adj_list[v]:
				if u != w:
					non_backtr[edge_dict[(u, v)]][edge_dict[(v, w)]] = 1

	# Returns the non-backtracking matrix (numpy int matrix) and edge dictionary
	return non_backtr, edge
