import math
import random
import numpy as np


def increase_opinion(t):
    # probability to increase opinion (move right)
    p0 = 0.01
    p = p0 * math.exp(-p0 * t)
    return p


def decrease_opinion(t):
    # probability to decrease opinion (move left)
    mu = 90
    s = 50
    q = 0.02 / (0.02 + math.exp(-(t - mu) / s))
    return q


def set_initial_opinion(N):
    op_out = np.random.choice([-1, 1], size=(N, 1))
    return op_out


def update_opinions(
    old_disease, old_opinion, N, opinion_network, disease_network, p, q, quarantine
):
    quarantine = np.zeros((N, 1))
    new_opinion = np.zeros((N, 1))

    for current_node_index in range(0, N):
        neighbors_opinion = get_neighbors(opinion_network, current_node_index)
        neighbors_disease = get_neighbors(disease_network, current_node_index)

        # calculate the number of infected nodes around the current node
        number_infected = 0
        for neighbor_index in neighbors_disease:
            if old_disease[neighbor_index] == 2:
                number_infected = number_infected + 1

        # calculate the phi parameter
        payoff_for = -0.1
        payoff_against = -number_infected
        payoff_net = payoff_for - payoff_against
        opinion_strength = 1
        phi = 1 / (1 + math.exp(-opinion_strength * payoff_net))

        random_neighbor_index = np.random.permutation(len(neighbors_opinion))[0]
        random_neighbor = neighbors_opinion[random_neighbor_index]

        # opinion exchange combinations
        if old_opinion[current_node_index] == old_opinion[random_neighbor]:
            if old_opinion[current_node_index] < 0 and random.random() < phi:
                new_opinion[current_node_index] = 1
            elif old_opinion[current_node_index] > 0 and random.random() < (1 - phi):
                new_opinion[current_node_index] = -1
            else:
                new_opinion[current_node_index] = old_opinion[current_node_index]

        elif old_opinion[current_node_index] < old_opinion[random_neighbor]:
            if random.random() < (0.5 * phi + 0.5 * p):
                new_opinion[current_node_index] = 1
            else:
                new_opinion[current_node_index] = old_opinion[current_node_index]

        elif old_opinion[current_node_index] > old_opinion[random_neighbor]:
            if random.random() < (0.5 * (1 - phi) + 0.5 * q):
                new_opinion[current_node_index] = -1
            else:
                new_opinion[current_node_index] = old_opinion[current_node_index]

    # update the quarantine status based on the newly formed opinions
    quarantine[new_opinion == 1] = 1

    return new_opinion


def get_neighbors(graph, node_index):
    return list(graph.neighbors(node_index))


def count_di(disease):
    total_count = [
        np.sum(disease == 0),
        np.sum(disease == 1),
        np.sum(disease == 2),
        np.sum(disease == 3),
    ]
    return total_count


def count_op(opinion):
    total_count = [np.sum(opinion == -1), np.sum(opinion == 1)]
    return total_count


def update_disease(
    old_state,
    opinion,
    N,
    disease_network,
    self_timer,
    beta,
    recovery_time,
    epsilon,
    quarantine,
):
    new_state = old_state

    list_of_infected = np.where(old_state == 2)[0]
    list_of_exposed = np.where(old_state == 1)[0]

    for index in range(0, len(list_of_infected)):
        i = list_of_infected[index]
        self_timer[i] += 1
        if self_timer[i] == recovery_time:
            new_state[i] = 3

        if quarantine[i] == 0:
            infect_neighbor(
                i, disease_network, beta, old_state, new_state, self_timer, quarantine
            )

        if random.random() < epsilon:
            new_state[i] = 2
            self_timer[i] = 0
            opinion[i] = 2
            quarantine[i] = 1

    return [new_state, self_timer]


def infect_neighbor(
    node, disease_network, beta, old_state, new_state, self_timer, quarantine
):
    neighbors = get_neighbors(node, disease_network)
    for j in range(0, len(neighbors)):
        neigh = neighbors[j]
        r = random.random()
        if r < beta and quarantine[neigh] == 0 and old_state[neigh] == 0:
            new_state[neigh] = 1
            self_timer[neigh] = 0
