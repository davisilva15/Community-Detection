import numpy as np


def BP_Inference(q, n, c, adj_list, criterium, t_max):
	"""
	Runs the belief propagation algorithm on a graph given by its adjacency list adj_list and (assumed) oracle parameters
	q (number of groups), n (proportion of nodes in each group) and c (probability of a pair of nodes from given groups to
	be joined by an edge multiplied by the number of nodes). Returns the infered group assignment and its free energy.
	"""
	# Number of nodes on the graph
	N = len(adj_list.keys())
	# The number of directed edges
	M = sum(len(neighbors) for neighbors in adj_list.values())

	# For each directed edge (u, v) we associate a code from 0 to 2|E| - 1
	edge_to_code = {}
	code_to_edge = []
	
	# For each directed edge we associate a "message" array of size q
	messages = np.random.rand(M, q)

	count = 0
	# For each node of the graph
	for u, neighbors in adj_list.items():
		# For each of u's neighbors
		for v in neighbors:
			# Each message has norm 1
			messages[count] /= np.sum(messages[count])
			edge_to_code[(u, v)] = count
			code_to_edge.append((u, v))
			count += 1

	# For each node we associate the marginal probability of it belonging to each group
	marg_prob = np.zeros((N, q))
	
	for u, neighbors in adj_list.items():
		# Compute the marginal probability array of node u
		p = marg_prob[u - 1]
		for v in neighbors:
			p += np.dot(c, messages[edge_to_code[(v, u)]])*messages[edge_to_code[(u, v)]]
		if len(neighbors) == 0:
			np.copyto(p, np.ones(q))
		p /= p.sum()

	# Initialization of the "external field" h
	h = np.zeros(q)
	for k in range(N):
		h += np.dot(c, marg_prob[k])
	h /= N

	# Measures the convergence of the messages
	conv = criterium + 1
	# Number of steps taken by the algorithm
	t = 0

	prod = np.zeros(q)
	old_message = np.zeros(q)
	old_marg_prob = np.zeros(q)
	while conv > criterium and t < t_max:
		t += 1
		conv = 0
		# Messages must be taken in random order to prevent cycles
		rand_order = np.random.permutation(M)
		for i in rand_order:
			# Vertices composing edge code i
			u, v = code_to_edge[i]
			# The current message
			curr_message = messages[i]
			# The current message before changes are made
			np.copyto(old_message, curr_message)
			
			# Updating current message
			np.copyto(prod, np.ones(q))
			for k in adj_list[u]:
				if k != v:
					prod *= np.dot(c, messages[edge_to_code[(k, u)]])
			np.copyto(curr_message, n*np.exp(-h)*prod)
			curr_message /= np.sum(curr_message)

			# Adding to conv the difference of new to the old message code i
			conv += np.linalg.norm(curr_message - old_message, 1)

			margv = marg_prob[v - 1]
			# The marginal probability array of node v before changes are made
			np.copyto(old_marg_prob, margv)
			# Updating the marginal probability array of node v
			margv *= np.dot(c, curr_message)/np.dot(c, old_message)
			margv /= np.sum(margv)

			# Updating the external field h
			h += np.dot(c, margv - old_marg_prob)/N

	# The estimated proportion of nodes on each group
	est_prop = np.sum(marg_prob, axis = 0)/N

	# The belief propagation estimate for the free energy and the estimated edge matrix
	f_BP, est_edges = free_energy(adj_list, N, q, est_prop, c, h, messages, edge_to_code)

	# An array containing the most probable group for each node
	groups = np.argmax(marg_prob, axis = 1) + np.ones(N, dtype = np.int8)

	return est_prop, est_edges, groups, f_BP


def free_energy(adj_list, N, q, n, c, h, messages, edge_to_code):
	"""
	Calculates the free energy associated to the parameters given and estimates the edge matrix
	"""
	f_BP = 0
	prod = n*np.exp(-h)
	prod_u = np.zeros(q)
	est_edges = np.zeros((q, q))
	
	# Only application of formulas from statistical physics
	for u in adj_list.keys():
		np.copyto(prod_u, prod)
		for v in adj_list[u]:
			muv = messages[edge_to_code[(u, v)]]
			mvu = messages[edge_to_code[(v, u)]]

			# All needed to compute the free energy
			cmvu = np.dot(c, mvu)
			prod_u *= cmvu
			Zuv = np.dot(muv, cmvu)
			# Needed to update the estimated edge matrix
			est_edges += np.outer(muv, mvu)/Zuv

			f_BP += np.log(Zuv)
		f_BP -= 2*np.log(np.sum(prod_u))
	f_BP /= (2*N)

	nn = np.outer(n, n)
	est_edges *= c/N
	# Average (directed) degree
	c_avg = np.sum(est_edges)
	# The estimated edge matrix
	est_edges /= nn

	# Final value of the Bethe free energy
	f_BP -= c_avg/2

	return f_BP, est_edges
