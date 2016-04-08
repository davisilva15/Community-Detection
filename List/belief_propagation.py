import numpy as np


def BP_Inference(q, n, c, adj_list, criterium, t_max):
	# Number of nodes on the graph
	N = len(adj_list.keys())

	# For each directed edge (u, v) we associate a code
	edge_to_code = {}
	code_to_edge = []
	# Interval of edge codes departing from each node
	ext_edges = np.zeros((N, 2), dtype = np.int)

	# For each directed edge we associate a "message" array of size q
	messages = []
	# For each node we associate the marginal probability of it belonging to each group
	marg_prob = np.zeros((N, q))

	count = 0
	# For each node of the graph
	for u in adj_list.keys():
		# First directed edge from u (if any) has code count
		ext_edges[u - 1][0] = count
		# For each of u's neighbors
		for v in adj_list[u]:
			# Each message is randomly initialized
			messages.append(rand_init(q))
			edge_to_code[(u, v)] = count
			code_to_edge.append((u, v))
			count += 1
		# Last directed edge from u has code count - 1
		ext_edges[u - 1][1] = count
		# Number of neighbors of node u
		nb_neighbors = count - ext_edges[u - 1][0]

		# Compute the marginal probability array of node u
		if nb_neighbors > 0:
			p = np.zeros(q)
			for i in range(ext_edges[u - 1][0], count):
				p += messages[i]
			p /= nb_neighbors
		else:
			p = np.ones(q)/p
		marg_prob[u - 1] = p

	# Initialization of the "external field" h
	h = np.zeros(q)
	for k in range(N):
		h += np.dot(c, marg_prob[k])
	h /= N

	conv = criterium + 1
	t = 0
	arr = np.arange(count)
	old_message = np.zeros(q)
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
			prod = np.ones(q)
			for k in adj_list[u]:
				if k != v:
					prod *= np.dot(c, messages[edge_to_code[(k, u)]])
			np.copyto(messages[i], n*np.exp(-h)*prod)
			messages[i] /= sum(messages[i])

			# Adding to conv the difference of new to the old message code i
			conv += np.linalg.norm(messages[i] - old_message)

			# The marginal probability array of node v before changes are made
			old_marg_prob = marg_prob[v - 1]
			# Updating the marginal probability array of node v
			marg_prob[v - 1] *= np.dot(c, messages[i])/np.dot(c, old_message)
			marg_prob[v - 1] /= sum(marg_prob[v - 1])

			# Updating the external field h
			h += np.dot(c, marg_prob[v - 1] - old_marg_prob)/N

	# The belief propagation estimate for the free energy
	f_BP = 0
	for u in adj_list.keys():
		prod_u = n*np.exp(-h)
		for v in adj_list[u]:
			Zuv = 0
			uv = edge_to_code[(u, v)]
			vu = edge_to_code[(v, u)]
			prod_u *= np.dot(c, messages[vu])
			for a in range(q):
				Zuv += c[a][a]*messages[uv][a]*messages[vu][a]
				for b in range(a + 1, q):
					Zuv += c[a][b]*(messages[uv][a]*messages[vu][b] + messages[uv][b]*messages[vu][a])
			f_BP += np.log(Zuv)
		f_BP -= 2*np.log(sum(prod_u))
	f_BP /= (2*N)

	# The average (directed) degree
	c_avg = 0
	for a in range(q):
		for b in range(q):
			c_avg += c[a][b]*n[a]*n[b]

	# Final value of the Bethe free energy
	f_BP -= c_avg/2

	
	groups = np.zeros(N, dtype = np.int)
	for u in range(N):
		groups[u] = np.argmax(marg_prob[u]) + 1	

	return groups


def rand_init(q):
	# A random array of size q with positive elements whose sum is 1
	r = np.random.rand(q)
	return r/sum(r)
