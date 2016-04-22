from random import random
import numpy as np


class Graph:
	def __init__(self, nb_vector, prob_matrix):
		"""
		Constructs a random graph given the number of nodes in each group and its edge probability matrix
		
		Attributes:
		-----------------
		nb_nodes: int
			The total number of nodes
		nb_groups: int
			The total number of groups
		adj_matrix: array[array[int]]
			The graph's adjacency matrix
		group: array[int]
			A map from each node to its group
		group_prop: array[float]
			An array showing the proportion of nodes in each group
		edge_prop: array[array[float]]
			The proportion of edges to all possible edges between groups
		"""
		# Total number of nodes on the graph
		self.nb_nodes = sum(nb_vector)
		# Total number of groups
		self.nb_groups = len(nb_vector)
		if len(prob_matrix) != self.nb_groups or len(prob_matrix[0]) != self.nb_groups:
			# If the matrix dimensions conflict with the total number of groups
			raise Exception("Conflicting data!")
		# The proportion of nodes in each group
		self.group_prop = nb_vector/self.nb_nodes

		# Constructs a vector indicating the biggest node key for each group
		last_nodes = np.copy(nb_vector)
		for i in range(1, self.nb_groups):
			last_nodes[i] += last_nodes[i-1]

		# Permutates the nodes so that there isn't an obvious relation between the nodes of each group
		dic = np.arange(self.nb_nodes)
		np.random.shuffle(dic)
		# Map from each node number to its group
		self.group = np.zeros(self.nb_nodes, dtype = np.int8)
		ind = 0
		for g in range(1, self.nb_groups + 1):
			while ind < last_nodes[g - 1]:
				self.group[dic[ind]] = g
				ind += 1

		# The graph's adjacency matrix
		self.adj_matrix = np.zeros((self.nb_nodes, self.nb_nodes), dtype = np.int8)
		# The actual proportion of edges (to all possible edges) between groups
		self.edge_prop = np.zeros((self.nb_groups, self.nb_groups))
		# Building the graph's edges
		for group1 in range(self.nb_groups):
			for group2 in range(group1, self.nb_groups):
				self.__build_edges(group1, group2, prob_matrix[group1][group2], last_nodes, dic)


	def __build_edges(self, g1, g2, p, last_nodes, dic):
		"""
		For each node1 in group g1 and node2 in group g2, an (undirected) edge is built between node1 and node2 with probability p
		"""
		# If there cannot be any edges between groups g1 and g2
		if p == 0:
			return

		# Number of nodes whose group number is smaller than g1
		shift1 = last_nodes[g1 - 1] if g1 > 0 else 0
		# Number of nodes whose group is g1
		n1 = last_nodes[g1] - shift1

		# If g1 and g2 are the same group
		if g1 == g2:
			# If all pairs of nodes from this group must be connected by edges
			if p == 1:
				for a in range(n1 - 1):
					for b in range(a + 1, n1):
						# Connects nodes number (shift1 + a) and (shift1 + b) by an edge
						self.adj_matrix[dic[shift1 + a]][dic[shift1 + b]] = 1
						self.adj_matrix[dic[shift1 + b]][dic[shift1 + a]] = 1
				# Computes the actual edge proportion matrix
				self.edge_prop[g1][g1] = 1
			
			# If the probability of each edge is in ]0, 1[
			else:
				# All possible edges are coded (from 1 to n1*(n1 - 1)/2)
				edge = 0
				a = 1
				while True:
					r = random()
					# Number of edges to skip
					k = int(np.log(1 - r)/np.log(1 - p))
					edge += k + 1

					# If all edges have already been decided
					if edge > (n1*(n1 - 1))//2:
						break

					# Decoding the edge number to the keys of its ending nodes
					while edge > a*n1 - (a*(a + 1))//2:
						a += 1
					b = edge - (a - 1)*n1 + (a*(a + 1))//2
					
					# Connects nodes number (shift1 + a - 1) and (shift1 + b - 1) by an edge
					self.adj_matrix[dic[shift1 + a - 1]][dic[shift1 + b - 1]] = 1
					self.adj_matrix[dic[shift1 + b - 1]][dic[shift1 + a - 1]] = 1
					self.edge_prop[g1][g1] += 2
				# Computes the actual edge proportion matrix
				self.edge_prop[g1][g1] /= n1*(n1 - 1)
		
		# If the groups g1 and g2 are different
		else:
			# Number of nodes whose group number is smaller than g2
			shift2 = last_nodes[g2 - 1]
			# Number of nodes whose group is g2
			n2 = last_nodes[g2] - shift2

			# If all pairs of nodes between these groups must be connected by edges
			if p == 1:
				for a in range(n1):
					for b in range(n2):
						# Connects nodes number (shift1 + a) and (shift2 + b) by an edge
						self.adj_matrix[dic[shift1 + a]][dic[shift2 + b]] = 1
						self.adj_matrix[dic[shift2 + b]][dic[shift1 + a]] = 1
				# Computes the actual edge proportion matrix		
				self.edge_prop[g1][g2] = 1
				self.edge_prop[g2][g1] = 1

			# If the probability of each edge is in ]0, 1[
			else:
				# All possible edges are coded (from 0 to n1*n2 - 1)
				edge = -1
				while True:
					r = random()
					# Number of edges to skip
					k = int(np.log(1 - r)/np.log(1 - p))
					edge += k + 1

					# If all edges have already been decided
					if edge >= n1*n2:
						break

					# Decoding the edge number to the keys of its ending nodes
					a = edge//n2
					b = edge%n2

					# Connects nodes number (shift1 + a) and (shift2 + b) by an edge
					self.adj_matrix[dic[shift1 + a]][dic[shift2 + b]] = 1
					self.adj_matrix[dic[shift2 + b]][dic[shift1 + a]] = 1
					self.edge_prop[g1][g2] += 1
				# Computes the actual edge probability matrix
				self.edge_prop[g1][g2] /= n1*n2
				self.edge_prop[g2][g1] = self.edge_prop[g1][g2]


	def write_in_file(self, file):
		"""
		Writes the graph generated with all its real attributes on the specified file
		"""
		f = open(file, 'w')

		f.write(str(self.nb_nodes) + '\n')
		f.write(str(self.nb_groups) + '\n')
		f.write(str(self.adj_matrix.tolist()) + '\n')
		f.write(str(self.group.tolist()) + '\n')
		f.write(str(self.group_prop.tolist()) + '\n')
		f.write(str(self.edge_prop.tolist()) + '\n')

		f.close()
