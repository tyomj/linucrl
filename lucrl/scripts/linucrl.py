import os
import numpy as np
import pandas as pd
import yaml
import random
import time
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')

project_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


from ..utils.coordinator import Coordinator
crd = Coordinator(project_dir)

from ..utils.logger import Logger
logger = Logger(project_dir, 'linucrl', 'linucrl.txt')

with open(os.path.join(crd.config.path, 'config.yaml')) as f:
    config = yaml.load(f)

def make_polynomial_features(recency_value, degree = config['linucrl']['d']):
    """
    recency_value: float
    degree: number of polynomial features. The target vector dimension is `degree + 1` cus each vector starts with 1 (recency_value^0).
    """
    return np.array([recency_value ** i for i in range(degree+1)])


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
        return make_polynomial_features(recency_value, config['linucrl']['d'])


def init_reward(state, action, init_reward_list):
    action_sequence = state.split('-')
    reward_of_given_action = [init_reward_list[i] for i, x in enumerate(action_sequence) if x == str(action)]

    return reward_of_given_action

def init_counter(state, action):
    action_sequence = state.split('-')
    return action_sequence.count(str(action))

class LinUCRL():

    def __init__(self, mdp):

        # mdp and global params
        self.mdp = mdp
        self.eval_it = config['linucrl']['eval_it']
        self.eval_steps = config['linucrl']['eval_steps']
        self.d = config['linucrl']['d'] # number of polynomial features
        self.window_size = config['mdp']['window_size'] # history len
        self.d_rate = config['linucrl']['d_rate'] # for VI
        self.vi_threshold = config['linucrl']['vi_threshold']  # for VI
        self.alpha_ = config['linucrl']['alpha']
        self.R_constant_ = config['linucrl']['R_constant']
        self.B_constant_ = config['linucrl']['B_constant']
        self.state = None # TODO seems redundant, it's better to use mdp attributes
        self.initial_reward = None
        self.global_reward = None # for visualization and metrics purposes
        self.max_rounds = config['linucrl']['max_rounds']
        self.t = 0

        # assign oracle greedy score for evaluation
        self.oracle_optimal_score = 0

        # parameters for C_t_a
        self.L_w = np.log(self.window_size) + 1
        self.L2_w = (1 - self.L_w ** (self.d + 1)) / (1 - self.L_w)

        # for each action define zero vector
        self.theta_vectors = {action: np.zeros(self.d + 1).reshape(-1, 1) for action in self.mdp.all_possible_actions}

        # define V matrices that then going to converge to X_transposed*X
        self.idn_matrix = np.identity(self.d + 1)
        self.lambda_ = config['linucrl']['linreg_reg_term']
        self.V_matrices = {action: (self.lambda_ * self.idn_matrix) for action in self.mdp.all_possible_actions}

        # other matrices
        self.X = {action: None for action in self.mdp.all_possible_actions}
        self.R = {action: np.array([]) for action in self.mdp.all_possible_actions}

        # and counters
        self.action_global_counter = {action: 0 for action in self.mdp.all_possible_actions}  # how many times we had seen the action since fit started (T_a in the paper)
        self.action_round_counter = {action: 0 for action in self.mdp.all_possible_actions}  # how many times we had seen the action since **round** started (Nu_a in the paper)

        self.policy = {state: random.sample(self.mdp.all_possible_actions, k=1)[0] for state in self.mdp.get_all_states()}
        self.value_function = {state: 0 for state in self.mdp.get_all_states()}

        # since cta value of an action uses in every value iteration step, it's better to cache them
        self.c_t_a_values = {action: 0 for action in self.mdp.all_possible_actions}

    def _all_vars_initializer(self):
        """
        Initialize all variables before VI loop.
        Updates class variables.

        :return: None
        """

        # Step 1
        # Reset an environment and observe an initial_state and initial_reward has obtained so far
        self.state, self.initial_reward = self.mdp.reset()
        self.global_reward = np.array(self.initial_reward)

        # Step 2
        # For each possible action define an initial feature matrix with shape == (window_size, d+1)
        # The first dimension of the matrix will then grow each time action `a` is chosen
        self.X = {action: get_features(self.state, action, construct_initial=True) for \
             action in self.mdp.all_possible_actions}

        # Step 3
        # When the matrix is defined we can easily update our identity matrices
        for action in self.mdp.all_possible_actions:
            self.V_matrices[action] = self.V_matrices[action] + np.dot(self.X[action].T, self.X[action])

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

        # updates
        self.X[action] = np.vstack((self.X[action], feature_vector))
        self.V_matrices[action] = self.V_matrices[action] + np.dot(feature_vector.reshape(-1, 1),
                                                                   feature_vector.reshape(-1, 1).T)
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
            self.state, reward, _ = self.mdp.step(action)
            # update all attributes
            self.update_class_instance_vars_in_episode(action, feature_vector, reward)
            self.action_global_counter[action] += 1

        return None

    def count_c_t_a(self, action):
        part1 = (1 + ((self.action_global_counter[action] + self.L2_w) / self.lambda_))
        part2 = self.mdp.nA * self.t**self.alpha_
        result = np.sqrt((self.d + 1) * np.log(part1 * part2))
        return self.R_constant_ * result + self.lambda_ ** 0.5 * self.B_constant_

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

    def value_iteration(self, V, thresh, discount_factor, reward_source='linucb'):
        """
        Value Iteration function.

        :param V: Value function to update
        :param thresh: We stop evaluation once our value function change is less than theta for all states.
        :param discount_factor: Gamma discount factor.
        :param reward_source: Defines which type of reward calculation to use.
                Takes two possible values:
                    `linucb` - get the reward from linear fucntion and UCB
                    `mdp` - get the reward directly from MDP (for oracle greedy VI)

        :return: A tuple (policy, V) of the optimal policy and the optimal value function.
        """
        logger.info('Starting value iteration')
        all_possible_acts = self.mdp.all_possible_actions

        def one_step_lookahead(state, V):
            """
            Observe one step ahead.

            :param state: The state to consider (int)
            :param V: The value to use as an estimator, Vector of length env.nS

            :return: A vector of length mdp.nA containing the expected value of each action.
            """

            A = pd.Series(np.zeros(len(all_possible_acts)), index=all_possible_acts)

            for action in A.index:
                if reward_source == 'linucb':
                    reward = self.linear_reward_function(state, action, theta=self.theta_vectors)
                    next_state = self.mdp.get_next_states(state, action, check_consistency=False)
                elif reward_source == 'mdp':
                    next_state, reward, _ = self.mdp.virtual_step(state, action)
                else:
                    assert "Wrong value for parameter `reward_source`"
                try:
                    A[action] += reward + discount_factor * V[next_state]
                except KeyError:
                    A[action] += reward

            return A

        while True:
            # Stopping condition
            delta = 0
            # Update each state...
            for s in V.keys():
                terminal = self.mdp.is_terminal(s)

                if not terminal:
                    A = one_step_lookahead(s, V)
                    best_action_value = A.max()
                    # Calculate delta across all states seen so far
                    delta = max(delta, np.abs(best_action_value - V[s]))
                    # Update the value function
                    V[s] = best_action_value
                else:
                    # If the state was terminal then there are no options to choose, hence there is no reward to observe
                    V[s] = 0
                    # Check if we can stop
            logger.info("delta is: {}".format(delta))
            if delta < thresh:
                break

        # Create a deterministic policy using the optimal value function
        policy = {}
        for s in tqdm(V.keys()):
            # One step lookahead to find the best action for this state
            A = one_step_lookahead(s, V)
            best_action = A.idxmax()
            # Always take the best action
            policy[s] = best_action
        logger.info('Value iteration finished')
        return policy, V

    def _get_oracle_optimal(self):
        logger.info('Calculating oracle reward ...')
        start = time.time()
        zero_value_function = {state: 0 for state in self.mdp.get_all_states()}
        policy_, value_function_ = self.value_iteration(V=zero_value_function, thresh=self.vi_threshold,
                                                                discount_factor=self.d_rate, reward_source='mdp')
        state_ = random.choice(tuple(self.mdp._df.state.tolist()))
        eval_cumulative_reward = []
        for _ in range(self.max_rounds):
            # Choose an action
            action_ = policy_[state_]
            # Get reward and next state
            state_, reward_, _ = self.mdp.virtual_step(state_, action_)
            # Append to evaluation reward
            eval_cumulative_reward.append(reward_)
        end = time.time()
        logger.info('Oracle reward calculated. Time elapsed: {}'.format(end - start))
        eval_cumulative_reward = np.array(eval_cumulative_reward)
        logger.info('Oracle sum reward: {}'.format(np.sum(eval_cumulative_reward)))

        return eval_cumulative_reward

    def fit(self, calc_oracle_optimal=True):
        """
        Fit LinUCRL model.

        :param calc_oracle_optimal: if `True` calculates oracle optima reward to compare with validation and plot results.
        :return:
        """

        if calc_oracle_optimal:
            self.oracle_optimal_score = self._get_oracle_optimal()

        self._all_vars_initializer()
        self._all_actions_initializer()

        # to avoid zero multiplication in c_t_a
        self.t = 1

        while self.t < self.max_rounds:
            logger.info("Round {}".format(self.t))

            assert self.state == self.mdp.current_state

            # Choose an action
            action = self.policy[self.state]
            # Obtain a feature vector
            feature_vector = get_features(self.state, action)
            # Use mdp to observe next state and real reward
            self.state, reward, _ = self.mdp.step(action)
            # Update all attributes
            self.update_class_instance_vars_in_episode(action, feature_vector, reward)

            #if after pulling the arm, round_counter exceeds global_counter then
            if self.action_round_counter[action] > self.action_global_counter[action]:

                start = time.time()
                # Update cta values
                self.c_t_a_values = {action: self.count_c_t_a(action) for action in self.mdp.all_possible_actions}
                #logger.info("Cta values: {} ".format(str(self.c_t_a_values)))
                # Run value iteration
                self.policy, self.value_function = self.value_iteration(V=self.value_function, thresh=self.vi_threshold, discount_factor=self.d_rate, reward_source='linucb')
                # and reassign global counter so that each time it's going to increase twice
                self.action_global_counter[action] += self.action_round_counter[action]
                # Reset action counter before new round is started
                self.action_round_counter = {action: 0 for action in self.mdp.all_possible_actions}
                end = time.time()
                logger.info('Policy updated. Time elapsed: {}'.format(end - start))
                #logger.info("Global counter state: {}".format(str(self.action_global_counter)))
            else:
                logger.info('Continue without update')

            """
            # Evaluate maximum reward every N rounds
            if self.t % self.eval_it == 0:
                print('Evaluation started')
                start = time.time()
                state_ = random.choice(tuple(self.mdp._df.state.tolist()))
                eval_cumulative_reward = []
                for _ in range(self.eval_steps):
                    # Choose an action
                    action_ = self.policy[state_]
                    # Get reward and next state
                    state_, reward_, _ = self.mdp.virtual_step(state_, action_)
                    # Append to evaluation reward
                    eval_cumulative_reward.append(reward_)
                end = time.time()
                print('Evaluation finished. Time elapsed: {}'.format(end - start))
                eval_cumulative_reward = np.sum(eval_cumulative_reward)
                print('Cumulative reward: {}'.format(eval_cumulative_reward))
                if count_oracle_optimal:
                    print('This is {0:.2f}% from oracle optimal result.'.format((eval_cumulative_reward / self.oracle_optimal_score) * 100))
            """

            self.t += 1

        if calc_oracle_optimal:

            path_to_save_plot = os.path.join(crd.logs.path, "plot.jpg")
            logger.info('Saving reward plot to: {}'.format(path_to_save_plot))
            g = pd.DataFrame(list(zip(self.global_reward, self.oracle_optimal_score)),
                         columns=['LinUCRL', 'Oracle_optimal']).plot()
            fig = g.get_figure()
            fig.savefig(path_to_save_plot)

        return None


