class ListNode:
	def __init__(self, graph_node, left = None, right = None):
		"""
		A doubly-linked list node whose data is a graph node
		"""
		self.node = graph_node
		self.left = left
		self.right = right


class LinkedList:
	def __init__(self):
		"""
		Initializes an empty doubly-linked list with a NIL sentinel
		"""
		self.NIL = ListNode(None)
		self.head = self.NIL
		self.tail = self.NIL

	def add_neighbor(self, neighbor):
		"""
		Adds a list node whose graph node is neighbor self's tail
		"""
		node = ListNode(neighbor, self.tail, self.NIL)
		self.tail.right = node
		self.tail = node
		if self.head is self.NIL:
			self.head = node