import numpy as np


def non_backtracking(adj_list):
	"""
	For a graph given by its adjacency list, returns its non-backtracking matrix and edge dictionary
	Time: O(|E|^2)		Space: O(|E|^2)
	"""
	# Maps linking each (directed) edge of the graph and a code from 0 to 2|E| - 1
	edge_to_code = {}
	code_to_edge = []
	count = 0
	for u in adj_list.keys():
		for v in adj_list[u]:
			edge_to_code[(u, v)] = count
			code_to_edge.append((u, v))
			count += 1

	# Construction of the non-backtracking matrix for this graph
	non_backtr = np.zeros((count, count), dtype = np.int8)
	for u in adj_list.keys():
		for v in adj_list[u]:
			uv = edge_to_code[(u, v)]
			for w in adj_list[v]:
				if u != w:
					vw = edge_to_code[(v, w)]
					non_backtr[uv][vw] = 1

	# Returns the non-backtracking matrix (numpy int matrix) and edge dictionary
	return non_backtr, code_to_edge
