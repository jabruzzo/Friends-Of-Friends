
# Import necessary packages
import math
import random
import numpy as np
import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt
import sets

# Constants
N_ROUNDS = 300

# Number of possible attitudes:
PARAM_N_ATTITUDES = 10

# Parameter controlling mean/variance over attitudes:
PARAM_Q = 0.5

# Number of agents:
PARAM_N = 100

# Number of initial connections per agent:
PARAM_D = 5

# Probability of accepting friendship:
PARAM_P = 0.05


# Useful functions that allow me to use lists like sets. These correspond to 
# the traditional mathematical definitions of union, intersection, and
# set difference:

# Returns A OR B
def union(l1, l2):

    return list(set(l1) | set(l2))



# Returns A AND B
def intersection(l1, l2):

	return list(set(l1) & set(l2))



# Returns (in A) AND NOT (in B)
def set_difference(l1, l2):

	return list(set(l1) - set(l2))



# Calculates the z-score between each node and its neighbors
def zscores(G):

	# Empty vector to hold all nodes' z-scores
	scores = []

	# For each node:
	for i in G.nodes():

		# Calculate the mean / sd of its neighbors' opinions
		m = []

		for n in G.neighbors(i):

			m.append(G.node[n]['attitude'])

		mu = np.mean(m)
		sigma = np.std(m)

		# Calculate z-score
		z = (G.node[i]['attitude'] - mu) / sigma

		# Add z-score to vector
		scores.append(z)

	# Returns the z-score vector
	return scores



# Function that returns a random boolean - TRUE if the agent accepts the 
# friend (which the agent does with probability PARAM_P), and FALSE if it
# does not
def accept_friend():

	p = random.random()

	if p < PARAM_P:

		return True

	return False



# Manages the simulation of the model
def simulate(G):

	# Selects 50% of the agents to update
	target_set = np.random.choice(G.nodes(), size = int(len(G.nodes())/2), replace = False)

	# An empty vector - will contain all new connections to be added to the
	# graph
	edges_to_add = []


	# For each node in the set to update:
	for t in target_set:

		# Grab the neighbors of the target node
		target_neighbors = G.neighbors(t)

		# Create an empty vector representing the "people you may know" feature
		# (i.e. a list of "friends of friends")
		people_you_may_know = []

		# For each of the target node's neighbors:
		for n in target_neighbors:

			# Grab the neighbors of the neighbors, but get rid of the node
			# itself, along with any nodes which are already neighbors with
			# the target node - this is the "people you may know" list
			neighbors_of_neighbors = G.neighbors(n)

			neighbors_of_neighbors = set_difference(neighbors_of_neighbors, [t])

			neighbors_of_neighbors = set_difference(neighbors_of_neighbors, target_neighbors)

			people_you_may_know = union(people_you_may_know, neighbors_of_neighbors)

		# Create a variable to hold the node with the greatest number of common
		# neighbors with the target and the number of common neighbors
		max_node = 0
		max_neighbors = 0

		# For each agent in "people you may know"
		for p in people_you_may_know:

			# How many neighbors in common does the agent have with the target
			# node?
			common_neighbors = list(nx.common_neighbors(G, t, p))

			n_common_neighbors = len(common_neighbors)

			# If p has more common neighbors than the agent currently listed
			# as having most common neighbors:
			if n_common_neighbors > max_neighbors:

				# List p as having the most common neighbors
				max_node = p
				max_neighbors = n_common_neighbors

			# If the target accepts p as a friend:
			if accept_friend():

				# Add the edge (t,p) to the edges to be added to the graph at 
				# the end of the time step
				sorted_edge = tuple(sorted((t, p)))

				edges_to_add = union(edges_to_add, [sorted_edge])

	# Add all new edges to the graph
	G.add_edges_from(edges_to_add)



# Function initializes the attitude / opinion values, which remain fixed in
# this version of the model. The values are initialized from a binomial
# distribution with parameters n = PARAM_N_ATTITUDES and p = PARAM_Q. 
def initialize_attitudes(G):

	attitudes = np.random.binomial(PARAM_N_ATTITUDES, PARAM_Q, size = PARAM_N)

	attitude_dict = dict(zip(G.nodes(), attitudes))

	nx.set_node_attributes(G, attitude_dict, name = 'attitude')	



# Main function that dictates overall program structure
def main():

	# Generate graph with PARAM_N nodes, each connected to PARAM_D nodes
	# selected uniformly at random
	G = nx.random_regular_graph(PARAM_D, PARAM_N)

	# Initialize the opinions / attitudes
	initialize_attitudes(G)

	# Empty vector to hold the average disagreement at each time step
	zmatrix = np.zeros((PARAM_N, N_ROUNDS), dtype = float)

	# For the desired number of time steps
	for i in range(N_ROUNDS):

		# Run the simulation and add to the list of average disagreements
		simulate(G)

		# Calculate the z-score for all the nodes in the graph
		zvec = zscores(G)
		zmatrix[:,i] = np.asarray(zvec, dtype = float)

	# Plot the z-scores for all the nodes at each time step
	for i in range(PARAM_N):
		plt.plot(range(N_ROUNDS), zmatrix[i,:])

	plt.show()



# Run the program
main()