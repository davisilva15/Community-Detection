import numpy as np
import graph_nodes


def non_backtracking(nodes):
	"""
	For a graph given by its nodes list, returns its non-backtracking matrix and edge dictionary
	Time: O(|E|^2)		Space: O(|E|^2)
	"""
	# Number of nodes on the graph
	n = len(nodes)

	# Dictionary giving for each (directed) edge a number from 0 to 2|E| - 1
	edge_dict = {}
	count = 0
	for u in nodes:
		for v in u.neighbors:
			edge_dict[(u.key, v.key)] = count
			count += 1

	# Construction of the non-backtracking matrix for this graph
	non_backtr = np.zeros((count, count), dtype = np.int)
	for u in nodes:
		for v in u.neighbors:
			for w in v.neighbors:
				if u != w:
					non_backtr[edge_dict[(u.key, v.key)]][edge_dict[(v.key, w.key)]] = 1

	# Returns the non-backtracking matrix (numpy int matrix) and edge dictionary
	return non_backtr, edge_dict
	
