import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from scipy.sparse import linalg
from read_matrix import read_file
from overlap import overlap


def Bethe_Hessian(adj_matrix, r):
	"""
	Given a real number r and the adjacency matrix of a graph, returns its Bethe-Hessian matrix
	"""
	N = len(adj_matrix)
	degree_matrix = np.zeros((N, N))
	degrees = np.sum(adj_matrix, axis = 1)
	for i, deg in enumerate(degrees):
		degree_matrix[i][i] = deg
	H = (r*r - 1)*np.eye(N) - r*adj_matrix + degree_matrix
	return H


def eigenvectors(adj_matrix, q_max):
	"""
	Given a graph's adjacency matrix and a maximum number of groups q_max to assign its nodes, returns the eigenvectors
	of the Bethe-Hessian matrix associated to the group structure
	"""
	# The number of nodes in the graph
	N = len(adj_matrix)
	# The average degree of a node in the graph
	c_avg = adj_matrix.sum()/(2*N)

	# Computes the Bethe-Hessian matrix with regularizer r = sqrt(c_avg)
	H1 = Bethe_Hessian(adj_matrix, np.sqrt(c_avg))
	# Computes the q_max smallest eigenvalues of H1 and their associated eigenvectors
	eig_val1, eig_vec1 = linalg.eigsh(H1, k = q_max, which = 'SA')

	# Indexes where the computed eigenvalues of H1 are negative
	ind1, = np.where(eig_val1 < 0)

	if len(ind1) == q_max:
		# If all q_max eigenvalues computed are negative (all groups are assortative)
		eig_vec = eig_vec1

	else:
		# Computes the Bethe-Hessian matrix with regularizer r = -sqrt(c_avg)
		H2 = Bethe_Hessian(adj_matrix, -np.sqrt(c_avg))
		# Computes the q_max - len(ind1) smallest eigenvalues of H2 and their associated eigenvectors
		eig_val2, eig_vec2 = linalg.eigsh(H2, k = q_max - len(ind1), which = 'SA')

		# Indexes where the computed eigenvalues of H2 are negative
		ind2, = np.where(eig_val2 < 0)

		if len(ind1) == 0:
			# If all groups are disassortative
			eig_vec = eig_vec2[:, ind2]

		elif len(ind2) == 0:
			# If all groups are assortative
			eig_vec = eig_vec1[:, ind1]

		else:
			# If there are both assortative and disassortative groups
			eig_vec = np.transpose(eig_vec1[:, ind1])
			np.append(eig_vec, np.transpose(eig_vec2[:, ind2]), axis = 0)
			eig_vec = np.transpose(eig_vec)

	# Returns a matrix containing the relevant eigenvectors
	return eig_vec


def BH_cluster_file(file):
	"""
	Given a file containing a graph's data, returns the normalized overlap between the group assignment infered by
	the Bethe-Hessian matrix spectral algorithm and the graph's true group assignment
	"""
	# Reads the graph's information
	N, q, adj_matrix, group, n, c = read_file(file)
	# The relevant eigenvectors for the spectral algorithm
	eig_vec = eigenvectors(adj_matrix, q)

	# Clusters the eigenvectors' coordinates in q groups using the K-means algorithm
	est = KMeans(n_clusters = q)
	est.fit(eig_vec)
	est_groups = est.labels_ + np.ones(N, dtype = np.int)

	# Returns the normalized overlap between the infered group assignment and the actual group assignment
	return overlap(N, q, n, est_groups, group)


def BH_cluster_matrix(adj_matrix, q_max):
	"""
	Given a graph's adjacency matrix and a maximum number of groups q_max to assign its nodes, returns the infered
	group assignment given by the Bethe-Hessian matrix spectral algorithm
	"""
	# The number of nodes in the graph
	N = len(adj_matrix)
	# The relevant eigenvectors for the spectral algorithm
	eig_vec = eigenvectors(adj_matrix, q_max)

	# Clusters the eigenvectors' coordinates in q groups using the K-means algorithm
	est = KMeans(n_clusters = len(eig_vec[0]))
	est.fit(eig_vec)
	est_groups = est.labels_ + np.ones(N, dtype = np.int)

	# Returns the infered group assignment
	return est_groups


def plot_graph(file):
	# Reads the graph's information
	N, q, adj_matrix, group, n, c = read_file(file)
	# Checks if the dimension of the embedding space is 2 or 3
	if q != 2 and q != 3:
		print('Cannot plot a multidimensional graph')
		return

	# The relevant eigenvectors for the spectral algorithm
	eig_vec = eigenvectors(adj_matrix, q)

	x = [np.array([np.zeros(int(round(N*ni))) for ni in n]) for _ in range(q)]
	# Separates the eigenvectors' coordinates according to their related groups
	cont = np.zeros(q, dtype = np.int)
	for i in range(N):
		for j in range(q):
			x[j][group[i] - 1][cont[group[i] - 1]] = eig_vec[i, j]
		cont[group[i] - 1] += 1

	if q == 2:
		# If there are two groups, plot the eigenvectors' coordinates on the plane
		plt.plot(x[0][0], x[1][0], 'ro', color = 'r')
		plt.plot(x[0][1], x[1][1], 'ro', color = 'b')
		plt.show()

	elif q == 3:
		# If there are three groups, plot the eigenvectors' coordinates on 3d space
		fig = plt.figure()
		ax = fig.add_subplot(111, projection = '3d')
		ax.scatter(x[0][0], x[1][0], x[2][0], 'ro', color = 'r')
		ax.scatter(x[0][1], x[1][1], x[2][1], 'ro', color = 'g')
		ax.scatter(x[0][2], x[1][2], x[2][2], 'ro', color = 'b')
		plt.show()
