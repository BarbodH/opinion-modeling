import math
import random
import numpy as np
import networkx as nx
from individual import Individual


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
    individuals: list[Individual],
    opinion_network: nx.Graph,
    disease_network: nx.Graph,
    p: float,
    q: float,
):
    for individual_index in range(0, len(individuals)):
        neighbors_opinion = get_neighbors(opinion_network, individual_index)
        neighbors_disease = get_neighbors(disease_network, individual_index)

        # calculate the number of infected nodes around the current node
        number_infected = 0
        for neighbor_index in neighbors_disease:
            if individuals[neighbor_index].disease == 2:
                number_infected = number_infected + 1

        # calculate the phi parameter
        payoff_for = -0.1
        payoff_against = -number_infected
        payoff_net = payoff_for - payoff_against
        opinion_strength = 1
        phi = 1 / (1 + math.exp(-opinion_strength * payoff_net))

        # opinion exchange combinations
        individual = individuals[individual_index]
        random_neighbor_index = neighbors_opinion[
            random.randint(0, len(neighbors_opinion) - 1)
        ]
        random_neighbor = individuals[random_neighbor_index]

        if individual.opinion == random_neighbor.opinion:
            if individual.opinion < 0 and random.random() < phi:
                individual.opinion = 1
            elif individual.opinion > 0 and random.random() < (1 - phi):
                individual.opinion = -1
        elif individual.opinion < random_neighbor.opinion:
            if random.random() < (0.5 * phi + 0.5 * p):
                individual.opinion = 1
        elif individual.opinion > random_neighbor.opinion:
            if random.random() < (0.5 * (1 - phi) + 0.5 * q):
                individual.opinion = -1


def get_neighbors(graph, node_index):
    return list(graph.neighbors(node_index))


def count_disease_states(individuals):
    total_S = sum(1 for individual in individuals if individual.disease == 0)
    total_I_A = sum(1 for individual in individuals if individual.disease == 1)
    total_I_S = sum(1 for individual in individuals if individual.disease == 2)
    total_R = sum(1 for individual in individuals if individual.disease == 3)
    return [total_S, total_I_A, total_I_S, total_R]


def count_opinion_states(individuals):
    total_against = sum(1 for individual in individuals if individual.opinion == -1)
    total_for = sum(1 for individual in individuals if individual.opinion == 1)
    return [total_against, total_for]


def update_disease(
    individuals: list[Individual],
    disease_network: nx.Graph,
    beta: float,
    recovery_time: int,
    epsilon: float,
):
    # new_state = old_state
    list_of_infected = [
        individual for individual in individuals if individual.disease == 2
    ]
    list_of_exposed = [
        individual for individual in individuals if individual.disease == 1
    ]

    for infected_individual in list_of_infected:
        infected_individual.increment_self_timer()
        if infected_individual.self_timer == recovery_time:
            infected_individual.recover()

        if infected_individual.quarantine == 0:
            infect_neighbor(infected_individual, individuals, disease_network, beta)

    for exposed_individual in list_of_exposed:
        exposed_individual.increment_self_timer()
        if exposed_individual.self_timer == recovery_time:
            exposed_individual.recover()

        if exposed_individual.quarantine == 0:
            infect_neighbor(exposed_individual, individuals, disease_network, beta)

        if random.random() < epsilon:
            exposed_individual.show_symptoms()


def infect_neighbor(
    individual: Individual,
    individuals: list[Individual],
    disease_network: nx.Graph,
    beta: float,
):
    neighbor_indices = get_neighbors(disease_network, individual.network_index)
    for neighbor_index in neighbor_indices:
        neighbor = individuals[neighbor_index]
        if (
            random.random() < beta
            and neighbor.quarantine == 0
            and neighbor.disease == 0
        ):
            neighbor.get_infected()
