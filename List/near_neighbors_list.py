def nearest_neighbors(adj_list, u, k):
	"""
	Given a graph by its adjacency list, a node u from this graph and a positive integer k, returns
	a dictionary that maps every node up to distance k from u to its distance from u
	"""
	# Checks if distance is positive
	if k < 0:
		raise IndexError("Distance must be positive")

	# Map from a node number to its distance from u
	distances = {u: 0}
	# Array containing nodes to visit
	to_visit = [u]

	# Current index on array to_visit
	ind = 0
	# Length of array to_visit
	last = 1

	# While there are still nodes to visit whose distance from u is smaller than k
	while ind < last and distances[to_visit[ind]] < k:
		v = to_visit[ind]
		dist = distances[v]
		for t in adj_list[v]:
			# If a neighbor of v hasn't yet been visited
			if t not in distances:
				to_visit.append(t)
				distances[t] = dist + 1
				last += 1
		ind += 1

	# Returns the map from a node number to its distance from u (if distance < k)
	return distances
