import numpy as np


def read_file(file):
	"""
	Reads and returns the graph's data written on a file
	"""
	# File where the graph data is writen
	f = open(file, 'r')

	# Number of nodes
	N = int(f.readline())
	# Number of groups
	q = int(f.readline())
	# Adjacency list
	adj_list = eval(f.readline())
	# Group assignment for each node
	group = np.array(eval(f.readline()), dtype = np.int8)
	# Proportion of nodes on each group
	n = np.array(eval(f.readline()))
	# Edge probability matrix times N
	c = np.array(eval(f.readline()))*N

	f.close()
	
	# Returns all data
	return N, q, adj_list, group, n, c
