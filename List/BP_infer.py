import numpy as np


def BP_Inference(q, n, c, adj_list, criterium, t_max):
	"""
	Runs the belief propagation algorithm on a graph given by its adjacency list adj_list and (assumed) oracle parameters
	q (number of groups), n (proportion of nodes in each group) and c (probability of a pair of nodes from given groups to
	be joined by an edge multiplied by the number of edges). Returns the infered group assignment and its free energy.
	"""
	# Number of nodes on the graph
	N = len(adj_list.keys())

	# For each directed edge (u, v) we associate a code from 0 to 2|E| - 1
	edge_to_code = {}
	code_to_edge = []
	
	# For each directed edge we associate a "message" array of size q
	messages = []

	count = 0
	# For each node of the graph
	for u in adj_list.keys():
		# For each of u's neighbors
		for v in adj_list[u]:
			# Each message is randomly initialized
			messages.append(rand_init(q))
			edge_to_code[(u, v)] = count
			code_to_edge.append((u, v))
			count += 1
	
	# For each node we associate the marginal probability of it belonging to each group
	marg_prob = np.zeros((N, q))
	
	for u in adj_list.keys():
		# List of nodes adjacent to u
		neighbors = adj_list[u]

		# Compute the marginal probability array of node u
		p = marg_prob[u - 1]
		for v in neighbors:
			p += np.dot(c, messages[edge_to_code[(v, u)]])*messages[edge_to_code[(u, v)]]
		if len(neighbors) == 0:
			np.copyto(p, np.ones(q))
		p /= sum(p)

	# Initialization of the "external field" h
	h = np.zeros(q)
	for k in range(N):
		h += np.dot(c, marg_prob[k])
	h /= N

	# Measures the convergence of the messages
	conv = criterium + 1
	# Number of steps taken by the algorithm
	t = 0
	# All codes of the directed edges
	arr = np.arange(count)

	prod = np.zeros(q)
	old_message = np.zeros(q)
	old_marg_prob = np.zeros(q)
	while conv > criterium and t < t_max:
		t += 1
		conv = 0
		# Messages must be taken in random order to prevent cycles
		np.random.shuffle(arr)
		for i in arr:
			# Vertices composing edge code i
			u, v = code_to_edge[i]
			# The message code i before changes are made
			np.copyto(old_message, messages[i])
			
			# Updating message code i
			np.copyto(prod, np.ones(q))
			for k in adj_list[u]:
				if k != v:
					prod *= np.dot(c, messages[edge_to_code[(k, u)]])
			np.copyto(messages[i], n*np.exp(-h)*prod)
			messages[i] /= sum(messages[i])

			# Adding to conv the difference of new to the old message code i
			conv += np.linalg.norm(messages[i] - old_message, 1)

			# The marginal probability array of node v before changes are made
			np.copyto(old_marg_prob, marg_prob[v - 1])
			# Updating the marginal probability array of node v
			marg_prob[v - 1] *= np.dot(c, messages[i])/np.dot(c, old_message)
			marg_prob[v - 1] /= sum(marg_prob[v - 1])

			# Updating the external field h
			h += np.dot(c, marg_prob[v - 1] - old_marg_prob)/N

	# Updating the array n
	np.copyto(n, np.sum(marg_prob, axis = 0)/N)

	# The belief propagation estimate for the free energy
	f_BP = free_energy(adj_list, N, q, n, c, h, messages, edge_to_code)

	# An array containing the most probable group for each node
	groups = np.argmax(marg_prob, axis = 1) + np.ones(N, dtype = np.int)

	return groups, f_BP


def rand_init(q):
	"""
	Returns a random array of size q with positive elements whose sum is 1
	"""
	r = np.random.rand(q)
	return r/sum(r)


def free_energy(adj_list, N, q, n, c, h, messages, edge_to_code):
	"""
	Calculates the free energy associated to the parameters given and updates matrix c
	"""
	f_BP = 0
	prod = n*np.exp(-h)
	prod_u = np.zeros(q)
	cmvu = np.zeros(q)
	c_new = np.zeros((q, q))
	
	# Only application of formulas from statistical physics
	for u in adj_list.keys():
		np.copyto(prod_u, prod)
		for v in adj_list[u]:
			uv = edge_to_code[(u, v)]
			vu = edge_to_code[(v, u)]

			# All needed to compute the free energy
			np.copyto(cmvu, np.dot(c, messages[vu]))
			prod_u *= cmvu
			Zuv = np.dot(np.transpose(messages[uv]), cmvu)
			# Needed to update the matrix c
			c_new += np.dot(messages[uv], np.transpose(messages[vu]))/Zuv

			f_BP += np.log(Zuv)
		f_BP -= 2*np.log(sum(prod_u))
	f_BP /= (2*N)

	nn = np.dot(n, np.transpose(n))
	c *= c_new/N
	# Average (directed) degree
	c_avg = np.sum(c)
	# Updated value of matrix c
	c /= nn

	# Final value of the Bethe free energy
	f_BP -= c_avg/2

	return f_BP
