from random import random
from math import log
import numpy as np


class Graph:
	def __init__(self, nb_vector, prob_matrix):
		"""
		Constructs a random graph given the number of nodes in each group and its edge probability matrix
		
		Attributes:
		-----------------
		last_group_nodes: array[int]
			The key of the largest node from each group
		nb_nodes: int
			The total number of nodes
		adj_matrix: array[array[int]]
			The graph's adjacency matrix
		"""
		# Total number of groups
		nb_groups = len(nb_vector)
		if len(prob_matrix) != nb_groups or len(prob_matrix[0]) != nb_groups:
			# If the matrix dimensions conflict with the total number of groups
			raise Exception("Conflicting data!")
		
		# Constructs a vector indicating the biggest node key for each group
		for i in range(1, nb_groups):
			nb_vector[i] += nb_vector[i-1]
		self.last_group_nodes = nb_vector
		# Total number of nodes on the graph
		self.nb_nodes = nb_vector[-1]
		# The graph's adjacency matrix
		self.adj_matrix = np.zeros((self.nb_nodes, self.nb_nodes), dtype = np.int)

		# Building the graph's edges
		for group1 in range(nb_groups):
			for group2 in range(group1, nb_groups):
				self.build_edges(group1, group2, prob_matrix[group1][group2])

	def build_edges(self, g1, g2, p):
		"""
		For each node1 in group g1 and node2 in group g2, an (undirected) edge is built between node1 and node2 with probability p
		"""
		# If there cannot be any edges between groups g1 and g2
		if p == 0:
			return

		# Number of nodes whose group number is smaller than g1
		shift1 = self.last_group_nodes[g1 - 1] if g1 > 0 else 0
		# Number of nodes whose group is g1
		n1 = self.last_group_nodes[g1] - shift1

		# If g1 and g2 are the same group
		if g1 == g2:
			# If all pairs of nodes from this group must be connected by edges
			if p == 1:
				for a in range(1, n1):
					for b in range(a + 1, n1 + 1):
						# Connects nodes number (shift1 + a) and (shift1 + b) by an edge
						self.adj_matrix[shift1 + a - 1][shift1 + b - 1] = 1
						self.adj_matrix[shift1 + b - 1][shift1 + a - 1] = 1
			
			# If the probability of each edge is in ]0, 1[
			else:
				# All possible edges are coded (from 1 to n1*(n1 - 1)/2)
				edge = 0
				a = 1
				while True:
					r = random()
					# Number of edges to skip
					k = int(log(1 - r)/log(1 - p))
					edge += k + 1

					# If all edges have already been decided
					if edge > (n1*(n1 - 1))//2:
						return

					# Decoding the edge number to the keys of its ending nodes
					while edge > a*n1 - (a*(a + 1))//2:
						a += 1
					b = edge - (a - 1)*n1 + (a*(a + 1))//2
					
					# Connects nodes number (shift1 + a) and (shift1 + b) by an edge
					self.adj_matrix[shift1 + a - 1][shift1 + b - 1] = 1
					self.adj_matrix[shift1 + b - 1][shift1 + a - 1] = 1
		
		# If the groups g1 and g2 are different
		else:
			# Number of nodes whose group number is smaller than g2
			shift2 = self.last_group_nodes[g2 - 1]
			# Number of nodes whose group is g2
			n2 = self.last_group_nodes[g2] - shift2

			# If all pairs of nodes between these groups must be connected by edges
			if p == 1:
				for a in range(1, n1 + 1):
					for b in range(1, n2 + 1):
						# Connects nodes number (shift1 + a) and (shift2 + b) by an edge
						self.adj_matrix[shift1 + a - 1][shift2 + b - 1] = 1
						self.adj_matrix[shift2 + b - 1][shift1 + a - 1] = 1

			# If the probability of each edge is in ]0, 1[
			else:
				# All possible edges are coded (from 0 to n1*n2 - 1)
				edge = -1
				while True:
					r = random()
					# Number of edges to skip
					k = int(log(1 - r)/log(1 - p))
					edge += k + 1

					# If all edges have already been decided
					if edge >= n1*n2:
						return

					# Decoding the edge number to the keys of its ending nodes
					a = edge//n2 + 1
					b = edge%n2 + 1

					# Connects nodes number (shift1 + a) and (shift2 + b) by an edge
					self.adj_matrix[shift1 + a - 1][shift2 + b - 1] = 1
					self.adj_matrix[shift2 + b - 1][shift1 + a - 1] = 1
