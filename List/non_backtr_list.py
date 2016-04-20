import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from scipy.sparse import linalg, csr_matrix
from overlap import overlap


def non_backtracking_sparse(adj_list):
	"""
	For a graph given by its adjacency list, returns its CSR sparse non-backtracking matrix and in-edges dictionary
	Time: O(|E|)		Space: O(|E|)
	"""
	# Dictionary from each (directed) edge of the graph to a code from 0 to 2|E| - 1
	edge_dict = {}
	# Dictionary from every node of the graph to the codes of its in-edges
	in_edges = {u: [] for u in adj_list.keys()}

	# Construction of both dictionaries
	count = 0
	for u in adj_list.keys():
		for v in adj_list[u]:
			edge_dict[(u, v)] = count
			in_edges[v].append(count)
			count += 1

	# Construction of the non-backtracking matrix for this graph
	row = []
	col = []
	for u in adj_list.keys():
		for v in adj_list[u]:
			uv = edge_dict[(u, v)]
			for w in adj_list[v]:
				if u != w:
					vw = edge_dict[(v, w)]
					row.append(uv)
					col.append(vw)

	# Returns the non-backtracking matrix (scipy sparse csr_matrix) and in-edges dictionary
	return csr_matrix((np.ones(len(row)), (row, col)), shape = (count, count)), in_edges


def NB_cluster(N, q, adj_list, group, n, plot_graph = False):
	# The graph's non-backtracking matrix and in-edges dictionary
	non_backtr, in_edges = non_backtracking_sparse(adj_list)

	# The non-backtracking matrix biggest eigenvalues (in module) and their associated eigenvectors
	eig_val, eig_vec = linalg.eigs(non_backtr, k = q, which = 'LR')
	# Indexes to sort the array of eigenvalues
	ind = np.argsort(eig_val.real)
	# The eigenvectors associated to the 2nd, 3rd, ..., qth biggest eigenvalues
	real_vec = eig_vec.real[:, ind[: q - 1]]
	
	# The matrix used on the embedding
	vecs = np.zeros((N, q - 1))
	for u in adj_list.keys():
		vecs[u - 1] = np.sum(real_vec[in_edges[u]], axis = 0)

	# If there are only two groups
	if q == 2:
		# Clusters according to the sign of vecs' coordinates
		est_groups = [int(vecs[i][0] >= 0) + 1 for i in range(N)]
	
	# More than two groups
	else:
		# Clusters vecs' rows in q groups using the K-means algorithm
		est = KMeans(n_clusters = q)
		est.fit(vecs)
		est_groups = est.labels_ + np.ones(N, dtype = np.int)

	# The normalized overlap between the infered group assignment and the actual group assignment
	ovlp = overlap(N, q, n, est_groups, group)

	# If the graph can be plotted and the user wants it to be plotted
	if plot_graph and q <= 4:
		# Separates vecs' rows according to their related groups
		x = [np.array([np.zeros(int(round(N*ni))) for ni in n]) for _ in range(q - 1)]
		cont = np.zeros(q, dtype = np.int)
		for i in range(N):
			for j in range(q - 1):
				x[j][group[i] - 1][cont[group[i] - 1]] = vecs[i, j]
			cont[group[i] - 1] += 1

		if q == 2:
			# If there are two groups, plot vecs' associated row for each node
			n0 = int(round(N*n[0]))
			n1 = int(round(N*n[1]))
			plt.plot(np.arange(n0), x[0][0], 'ro', color = 'r')
			plt.plot(np.arange(n0, n0 + n1), x[0][1], 'ro', color = 'b')
			plt.show()

		elif q == 3:
			# If there are three groups, plot vecs' rows on the plane
			plt.plot(x[0][0], x[1][0], '^', color = 'r')
			plt.plot(x[0][1], x[1][1], 'x', color = 'b')
			plt.plot(x[0][2], x[1][2], '+', color = 'y')
			plt.show()

		elif q == 4:
			# If there are four groups, plot vecs' rows on 3d space
			fig = plt.figure()
			ax = fig.add_subplot(111, projection = '3d')
			ax.scatter(x[0][0], x[1][0], x[2][0], marker = '^', c = 'r')
			ax.scatter(x[0][1], x[1][1], x[2][1], marker = 'x', c = 'g')
			ax.scatter(x[0][2], x[1][2], x[2][2], marker = '+', c = 'b')
			ax.scatter(x[0][3], x[1][3], x[2][3], marker = 'o', c = 'y')
			plt.show()

	# Returns the calculated overlap
	return ovlp
