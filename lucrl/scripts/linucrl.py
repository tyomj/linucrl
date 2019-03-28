import os
import numpy as np
import pandas as pd
import yaml
import random
import time
from tqdm import tqdm
from itertools import product

from functools import lru_cache

project_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

from lucrl.utils.coordinator import Coordinator
crd = Coordinator(project_dir)

from lucrl.utils.logger import Logger
logger = Logger(project_dir, 'linucrl', 'linucrl.txt')

with open(os.path.join(crd.config.path, 'config.yaml')) as f:
    config = yaml.load(f)

from lucrl.scripts.base import Base

@lru_cache(maxsize=None)
def make_polynomial_features(recency_value, degree = config['linucrl']['d']):
    """
    recency_value: float
    degree: number of polynomial features. The target vector dimension is `degree + 1` cus each vector starts with 1 (recency_value^0).
    """
    return np.array([recency_value ** i for i in range(degree+1)])

@lru_cache(maxsize=None)
def get_features(state, action, construct_initial=False):
    """
    If `construct_initial` is `True` returns initial matrix of features with shape (window_size, d+1).
    Else returns feature vector for a planned action `a` given state vector with len == window_size.
    """

    # make an inverted list from state string
    list_of_states = [int(x) for x in state.split(config['mdp']['state_delimiter'])][::-1]
    # take all the indexes with a particular action in the past
    indexes = [i + 1 for i, x in enumerate(list_of_states) if x == action]
    # count recency value for an action `a`
    recency_value = np.sum([1 / x for x in indexes])

    # if u need to obtain an initial features
    if construct_initial:
        # make an empty list of recency values
        recency_values = []

        # define a closure function
        def get_initial_features(list_states):
            if len(list_states) > 0:
                indexes_ = [i + 1 for i, x in enumerate(list_states) if x == action]
                recency_value_ = np.sum([1 / x for x in indexes_])
                recency_values.append(recency_value_) # out of scope - not the best option TODO fix this
                return get_initial_features(list_states[1:])

        # recursively obtain all of the recency values from the past
        get_initial_features(list_of_states)
        # and construct a matrix with poly features
        feature_matrix = np.stack([make_polynomial_features(x, config['linucrl']['d']) for x in recency_values])

        # for initial matrix
        return feature_matrix
    else:
        # for vector
        feature_vector = make_polynomial_features(recency_value, config['linucrl']['d'])
        return feature_vector

def init_reward(state, action, init_reward_list):
    action_sequence = state.split('-')
    reward_of_given_action = [init_reward_list[i] for i, x in enumerate(action_sequence) if x == str(action)]

    return reward_of_given_action

def init_counter(state, action):
    action_sequence = state.split('-')
    return action_sequence.count(str(action))

class LinUCRL(Base):

    def __init__(self, mdp):
        super().__init__(mdp)

        self.initial_reward = None

        # parameters for C_t_a
        self.L_w = np.log(self.window_size) + 1
        self.L2_w = (1 - self.L_w ** (self.d + 1)) / (1 - self.L_w)

        # for each action define zero vector
        self.theta_vectors = {action: np.zeros(self.d + 1).reshape(-1, 1) for action in self.mdp.all_possible_actions}

        # define V matrices that then going to converge to X_transposed*X
        self.idn_matrix = np.identity(self.d + 1)
        self.lambda_ = config['linucrl']['linreg_reg_term']
        #self.V_matrices = {action: (self.lambda_ * self.idn_matrix) for action in self.mdp.all_possible_actions}
        self.V_matrices = {action: (self.alpha_ * self.idn_matrix) for action in self.mdp.all_possible_actions}

        # other matrices
        self.X = {action: None for action in self.mdp.all_possible_actions}
        self.R = {action: np.array([]) for action in self.mdp.all_possible_actions}

        # and counters
        self.action_global_counter = {action: 0 for action in self.mdp.all_possible_actions}  # how many times we had seen the action since fit started (T_a in the paper)
        self.action_round_counter = {action: 0 for action in self.mdp.all_possible_actions}  # how many times we had seen the action since **round** started (Nu_a in the paper)
        self.actions_taken = []

        # since cta value of an action uses in every value iteration step, it's better to cache them
        self.c_t_a_values = {action: 0 for action in self.mdp.all_possible_actions}

        self.all_states_combinations = list(product(*[list(self.mdp.all_possible_actions) for _ in range(self.window_size)]))
        self.all_states_combinations = set(['-'.join([str(x_) for x_ in x]) for x in self.all_states_combinations])

        self.policy = {state: 0 for state in self.all_states_combinations}
        self.value_function = {state: 0 for state in self.all_states_combinations}

    def _all_vars_initializer(self):
        """
        Initialize all variables before VI loop.
        Updates class variables.

        :return: None
        """

        # Step 1
        # Reset an environment and observe an initial_state and initial_reward has obtained so far
        self.state, self.initial_reward = self.mdp.reset()
        #self.global_reward = np.array(self.initial_reward)

        # Step 2
        # For each possible action define an initial feature matrix with shape == (window_size, d+1)
        # The first dimension of the matrix will then grow each time action `a` is chosen
        self.X = {action: get_features(self.state, action, construct_initial=True) for \
             action in self.mdp.all_possible_actions}

        # Step 3
        # When the matrix is defined we can easily update our identity matrices
        for action in self.mdp.all_possible_actions:
            m_upd = np.dot(self.X[action].T, self.X[action])
            self.V_matrices[action] = self.V_matrices[action] + m_upd

        # Step 4
        # Init theta vectors for each action based on reward observed so far
        for action in self.mdp.all_possible_actions:
            self.theta_vectors[action] = np.dot(np.linalg.inv(self.V_matrices[action]),
                                   np.dot(self.X[action].T, self.initial_reward.reshape(-1, 1)))

        # Step 5
        # Init reward
        for action in self.mdp.all_possible_actions:
            self.R[action] = np.append(self.R[action], init_reward(self.state, action, self.initial_reward))

        # Step 6
        # Update global counter
        for action in self.mdp.all_possible_actions:
            self.action_global_counter[action] = self.action_global_counter[action] + init_counter(self.state, action)

        return None

    def update_class_instance_vars_in_episode(self, action, feature_vector, reward):
        """
        Updates all the variables in one step.

        :param action: action has been taken following a policy
        :param feature_vector: feature vector has been constructed
        :param reward: real reward, observed from MDP

        :return: None
        """

        if reward != 0:
            # In the original implementation update uses only for real state with reward
            self.X[action] = np.vstack((self.X[action], feature_vector))

            matrix_upd = feature_vector.reshape(1,-1).T.dot(feature_vector.reshape(1,-1))
            self.V_matrices[action] = self.V_matrices[action] + matrix_upd
            self.R[action] = np.append(self.R[action], reward)
            self.global_reward = np.append(self.global_reward, reward)
            self.action_round_counter[action] += 1


        return None

    def _all_actions_initializer(self):
        """
        Helper function to fill up a history for each action at least once,
        if not all the possible actions was executed in a randomly chosen initial state from MDP.

        Updates class variables.

        :return: None
        """
        # for initialization we will use real reward values to compute theta
        not_visited_actions = [a for a, counter_value in self.action_round_counter.items() if counter_value == 0]

        for action in not_visited_actions:
            # get recency-based polynomial feature vector
            feature_vector = get_features(self.state, action)
            # use mdp to observe next state and real reward
            self.state, reward, _ = self.mdp.step(self.state, action)
            # update all attributes
            self.update_class_instance_vars_in_episode(action, feature_vector, reward)
            self.action_global_counter[action] += 1

        return None

    def count_c_t_a(self, action):
        part1 = (1 + ((self.action_global_counter[action] + self.L2_w) / self.lambda_))
        part2 = self.mdp.nA * (self.t+1)**self.alpha_
        result = np.sqrt((self.d + 1) * np.log(part1 * part2))
        conf = result + self.lambda_ ** 0.5

        # from original
        if self.action_global_counter[action] == 0:
            conf *= 10.0
        else:
            conf *= 0.01

        return conf

    def linear_reward_function(self, state, action, theta):

        # get linear reward
        feature_vector = get_features(state, action)
        feature_vector_transposed = feature_vector.reshape(-1, 1).T
        linear_reward = np.dot(theta[action], feature_vector_transposed)[0][0]

        # get UCB term
        c_t_a = self.c_t_a_values[action]
        vector_term = np.sqrt(np.dot(feature_vector_transposed, np.linalg.solve(self.V_matrices[action],
                                                                                feature_vector.reshape(-1, 1))))
        # total
        optimistic_reward = linear_reward + c_t_a*vector_term

        return optimistic_reward

    def one_step_lookahead(self, _state, _action, _V, _precomputed_rewards, _all_possible_states, _discount_factor):
        """
        Observe one step ahead.

        :param _state: The state to consider (int)
        :param _action: The state to consider (int)
        :param _V: The value to use as an estimator, Vector of length env.nS
        :param _precomputed_rewards: The value to use as an estimator, Vector of length env.nS
        :param _all_possible_states: The value to use as an estimator, Vector of length env.nS
        :param _discount_factor: The value to use as an estimator, Vector of length env.nS

        :return: A vector of length mdp.nA containing the expected value of each action.
        """
        r = 0
        _reward = _precomputed_rewards[_state][_action]
        _next_state = self.mdp.get_next_states(_state, _action, check_consistency=False)
        if _next_state in _all_possible_states:
            r += _reward + _discount_factor * _V[_next_state]
        else:
            r += _reward

        return r

    def value_iteration(self, V, discount_factor):
        """
        Value Iteration function.

        :param V: Value function to update
        :param discount_factor: Gamma discount factor.

        :return: A tuple (policy, V) of the optimal policy and the optimal value function.
        """
        logger.info('Starting value iteration')
        all_possible_acts = self.mdp.all_possible_actions
        all_possible_states = self.all_states_combinations

        logger.info('Precomputing reward')
        # compute rewards once(!!!) for optimization purposes (taken from original)
        _rewards = dict.fromkeys(list(V.keys()))
        for state in tqdm(set(list(V.keys()))):
            _rewards[state] = dict.fromkeys(all_possible_acts)
            for action in all_possible_acts:
                reward = self.linear_reward_function(state, action, theta=self.theta_vectors)
                _rewards[state][action] = reward

        shuffled_states = list(V.keys())
        random.shuffle(shuffled_states)

        logger.info('Value iteration')

        for i_ in tqdm(range(self.vi_iters)):
            # Update each state...
            for s in shuffled_states:
                # Update the value function
                V[s] = max(list(map(lambda a: self.one_step_lookahead(_state=s,
                                                                      _action=a,
                                                                      _V=V,
                                                                      _precomputed_rewards=_rewards,
                                                                      _all_possible_states=all_possible_states,
                                                                      _discount_factor=discount_factor),
                                    all_possible_acts)))

        logger.info('Obtain the best policy so far')
        # Create a deterministic policy using the optimal value function
        policy = {}
        for s in tqdm(V.keys()):
            # One step lookahead to find the best action for this state
            A_dict = {a: self.one_step_lookahead(_state=s,
                                                 _action=a,
                                                 _V=V,
                                                 _precomputed_rewards=_rewards,
                                                 _all_possible_states=all_possible_states,
                                                 _discount_factor=discount_factor)[0][0] for a in all_possible_acts}
            A_ = pd.Series(A_dict)
            best_action = A_.idxmax()
            # Always take the best action
            policy[s] = best_action
        logger.info('Value iteration finished')
        return policy, V

    def fit(self):
        """
        Fit LinUCRL model.

        :return:
        """

        self._all_vars_initializer()
        self._all_actions_initializer()

        # to avoid zero multiplication in c_t_a
        self.t = np.sum(list(self.action_global_counter.values()))

        zero_value_function = {state: 0 for state in self.all_states_combinations}

        while self.t < self.max_rounds:
            logger.info("Round {}".format(self.t))


            if not self.state == self.mdp.current_state:

                self.state, _ = self.mdp.reset()

            # Choose an action
            #print("state: {}".format(self.state))
            action = self.policy[self.state]

            #print("action: {}".format(action))
            # Obtain a feature vector
            feature_vector = get_features(self.state, action)
            #print("feature_vector: {}".format(feature_vector))
            # Use mdp to observe next state and real reward
            self.state, reward, _ = self.mdp.step(self.state, action)


            if reward != 0:
                self.actions_taken.append(action)

                # Update all attributes
                self.update_class_instance_vars_in_episode(action, feature_vector, reward)
                self.t += 1

                # if after pulling the arm, round_counter exceeds global_counter then
                if (self.action_round_counter[action] > self.action_global_counter[action]) and (
                    self.action_round_counter[action] > 1):

                    start = time.time()
                    # Update cta values
                    self.c_t_a_values = {action: self.count_c_t_a(action) for action in self.mdp.all_possible_actions}
                    #print("c_t_a_values: {}".format(self.c_t_a_values))
                    # and reassign global counter so that each time it's going to increase twice
                    self.action_global_counter[action] += self.action_round_counter[action]
                    # Reset action counter before new round is started
                    self.action_round_counter = {action: 0 for action in self.mdp.all_possible_actions}
                    # Run value iteration
                    self.policy, self.value_function = self.value_iteration(V=self.value_function,
                                                                            discount_factor=self.d_rate)
                    end = time.time()
                    logger.info('Policy updated. Time elapsed: {}'.format(end - start))
                    #print("Global counter state:  %s", self.action_global_counter)
                else:
                    logger.info('Continue without update')

            else:
                # chill out
                pass

        return None


