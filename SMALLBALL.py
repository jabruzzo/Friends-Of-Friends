
# Import necessary packages
import math
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sets


# Constants
N_ROUNDS = 400

PARAM_ATTITUDES = [0,1,2,3,4,5,6,7,8,9]
PARAM_DIST = [.1,.1,.1,.1,.1,.1,.1,.1,.1,.1]

PARAM_N = 100 # number of agents
PARAM_D = 5 # number of ties
PARAM_P = 0.05 # the probability that the node will accept
PARAM_Q = 0.01 #


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


# Calculates the average disagreement statistic that I'm using to measure
# convergence
'''
change average_disagreement to
1. SD
2. Z score
'''

def average_disagreement(G):
	# Empty vector to hold each node's average disagreement value
	disagreements = []
	# For each node:
	for i in G.nodes():

		d = 0

		# Calculate the average opinion difference between it and its
		# neighbors and append it to the vector
		for n in G.neighbors(i):

			d += abs(G.node[i]['attitude'] - G.node[n]['attitude'])

		disagreements.append(float(d) / len(list(G.neighbors(i))))

	# Return the average of the average opinion differences
	return np.mean(disagreements)


# Function that returns a random boolean - TRUE if the agent accepts the
# friend (which the agent does with probability PARAM_P), and FALSE if it
# does not
def accept_friend():

	p = random.random()

	if p < PARAM_P:

		return True

	return False

# Returns a random boolean representing whether or not the target "sees"
# the "person it may know"
def sees_agent(pr):

	p = random.random()

	if p < pr:

		return True

	return False



# Returns the probability that the "person you may know" is seen by the target
# It's just the geometric distribution with parameter PARAM_Q. This is written
# such that people with more friends in common are more likely to be "seen"
# by the target
def pr_visible_function(n_common_neighbors):

	return PARAM_Q * ((1 - PARAM_Q) ** n_common_neighbors)



# Manages the simulation of the model
def simulate(G):

	# Selects 50% of the agents to update
	target_set = np.random.choice(G.nodes(), size = len(G.nodes())/2, replace = False)

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

		# For each agent in "people you may know"
		for p in people_you_may_know:

			# How many neighbors in common does the agent have with the target
			# node?
			common_neighbors = list(nx.common_neighbors(G, t, p))

			n_common_neighbors = len(common_neighbors)

			# Does the target node see this "person it may know"? This
			# function is written such that people with more friends in
			# common are "seen" more frequently
			pr_shown_to_target = pr_visible_function(n_common_neighbors)

			# If the target sees the agent and accepts it as a friend:
			if sees_agent(pr_shown_to_target) and accept_friend():

				# Add the edge to the edges to be added to the graph at the
				# end of the time step
				sorted_edge = tuple(sorted((t, p)))

				edges_to_add = union(edges_to_add, [sorted_edge])

	# Add all new edges to the graph
	G.add_edges_from(edges_to_add)



# Function initializes the attitude / opinion values, which remain fixed in
# this version of the model
def initialize_attitudes(G):

	# Right now, I'm using the following heuristic to fix the attitudes: I
	# pick some reference node at random. Then the opinions of all other
	# nodes are the length of the shortest path from those nodes to the
	# reference node. This is just a placeholder until we get the "51%"
	# issue straightened out

	reference = np.random.choice(list(G.nodes()))

	attitudes = []

	for i in G.nodes():

		sp = nx.shortest_path_length(G, reference, i)

		attitudes.append(sp)

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
	avg_disagreement_vals = []

	# For the desired number of time steps
	for i in range(N_ROUNDS):


		# Run the simulation and add to the list of average disagreements
		simulate(G)

		avg_disagreement_vals.append(average_disagreement(G))

	# Plot the average disagreement sample path after the desired number
	# of time steps
	plt.plot(range(N_ROUNDS), avg_disagreement_vals)
	plt.show()



# Run the program
main()
