import random
import yaml
import os
from functools import lru_cache

project_dir = os.path.normpath(os.path.dirname(os.path.abspath(__file__)) + os.sep + os.pardir)

from lucrl.utils.coordinator import Coordinator
crd = Coordinator(project_dir)

from lucrl.utils.logger import Logger
logger = Logger(project_dir, 'ml_to_df', 'ml_to_df.txt')

with open(os.path.join(crd.config.path, 'config.yaml')) as f:
    config = yaml.load(f)

def check_mdp(mdp):
    """
    Simple function to print meta info and check stationarity.
    Consciously defined outside of the class.

    :param mdp: an object of MDP class
    :return:
    """
    logger.info("Total states: {}".format(mdp.nS))
    logger.info("Total actions: {}".format(mdp.nA))
    logger.info("mdp.get_all_possible_actions: {}".format(mdp.all_possible_actions))
    try:
        possible_actions = mdp.get_possible_actions(mdp.current_state)
        logger.info("mdp.get_possible_actions: ", possible_actions)
    except Exception as e:
        logger.info("Failed to call {}: {}".format('mdp.get_possible_actions', e))
    try:
        next_state = mdp.get_next_states(mdp.current_state, possible_actions[0])
        logger.info("mdp.get_next_states: {}".format(next_state))
    except Exception as e:
        logger.info("Failed to call {}: {}".format('mdp.get_next_states', e))
    try:
        reward = mdp.get_reward(state=mdp.current_state,
                                                  action=possible_actions[0],
                                                  next_state=next_state)
        logger.info("mdp.get_reward: {}".format(reward))
    except Exception as e:
        logger.info("Failed to call {}: {}".format('mdp.get_reward', e))

    return True

class MDP:
    def __init__(self, df, state_delimiter=config['mdp']['state_delimiter'], initial_state=None, initial_reward=None, random_state=0):
        """
        Defines an MDP.
        :param states: a list of strings. Each string represents previously taken acions separated by delimeter '-' with action len == window size
        Example: '1-0-1' - is a state with window size == 3.

        :param state_delimiter: str, a symbol to specify a delimiter
        :param initial_state: str, optional. A symbol to specify a delimiter
        :param initial_reward: list, optional. A list of ints to specify previously obtained reward for the state with len == 'window_size'
        By default, picks initial state at random.
        }
        """
        self._df = df
        self.state_delimiter = state_delimiter
        self._initial_state = initial_state
        self._initial_reward = initial_reward
        self.current_state = None
        self.nS = self._df.state.nunique() # number of unique states
        self.nA = self._df.genres.nunique() # number of unique actions
        self.all_possible_actions = self._get_all_possible_actions() # a tuple of all actions (refers to [K] in the paper)
        self.reset()

        # for optimization
        self.state_reward_dict = self._df[['state', 'reward']].set_index('state').to_dict(orient='index')
        self.state_reward_dict = {k: v['reward'] for k, v in self.state_reward_dict.items()}
        random.seed(random_state)

    def get_all_states(self):
        """ return a tuple of all possible states """
        return tuple(self._df.state)

    def _get_all_possible_actions(self):
        """ return a tuple of all possible states """
        return tuple(self._df.genres.unique())

    @lru_cache(maxsize=None)
    def get_possible_actions(self, state):
        """ return a tuple of possible actions in a given state """
        potential_new_state_start = '-'.join(state.split(self.state_delimiter)[1:])
        potential_states = self._df[self._df.state.str.startswith(potential_new_state_start)].state.tolist()
        if potential_states:
            possible_actions = [int(x.split('-')[-1]) for x in potential_states]
            return tuple(possible_actions)
        else:
            return tuple()

    @lru_cache(maxsize=None)
    def is_terminal(self, state):
        """ return True if state is terminal or False if it isn't """
        if self.get_possible_actions(state):
            return False
        else:
            return True

    @lru_cache(maxsize=None)
    def get_next_states(self, state, action, check_consistency=False):
        """ return a dictionary of {next_state1 : P(next_state1 | state, action), next_state2: ...} """
        if check_consistency:
            assert action in self.get_possible_actions(
                state), "cannot do action %s from state %s" % (action, state)
        next_state = '-'.join(state.split(self.state_delimiter)[1:]) + '-' + str(action)
        return next_state

    def get_transition_prob(self, state, action, next_state):
        """ return P(next_state | state, action) """
        return self.get_next_states(state, action).get(next_state, 0.0)

    @lru_cache(maxsize=None)
    def get_reward(self, state, action, next_state):
        """ return the reward you get for taking action in state and landing on next_state"""
        if action in self.get_possible_actions(state):
            try:
                #reward = self._df[self._df.state == next_state].reward.values[0]
                reward = self.state_reward_dict[next_state]
                return reward

            except Exception as ex:
                logger.debug("Exception: {}, state: {}, action: {}".format(ex, state, action))
                return 0

        else:
            logger.debug("cannot do action %s from state %s" % (action, state))
            return 0


    def reset(self):
        """ reset the game, return the initial state"""
        if self._initial_state is None:
            self.current_state = random.choice(
                tuple(self._df.state.tolist()))
        elif self._initial_state in tuple(self._df.state.tolist()):
            self.current_state = self._initial_state
        elif callable(self._initial_state):
            self.current_state = self._initial_state()
        else:
            raise ValueError(
                "initial state %s should be either a state or a function() -> state" % self._initial_state)
        self._initial_reward = self._df[self._df.state == self.current_state].initial_reward.values[0][0]

        return self.current_state, self._initial_reward

    @lru_cache(maxsize=None)
    def virtual_step(self, state, action):
        """
        Take an action from any state u like without updating current state.

        :param state:
        :param action:
        :return: next_state, reward, is_done
        """

        next_state = self.get_next_states(state, action, check_consistency=False)
        reward = self.get_reward(state, action, next_state)
        #is_done = self.is_terminal(next_state)
        return next_state, reward, None #, is_done

    @lru_cache(maxsize=None)
    def step(self, state, action):
        """
        Take an action for current state, update current state.

        :param action:
        :param state:
        :return: next_state, reward, is_done
        """
        next_state = self.get_next_states(state, action)
        reward = self.get_reward(state, action, next_state)
        is_done = self.is_terminal(next_state)
        self.current_state = next_state
        return next_state, reward, is_done

    def render(self):
        logger.info("Currently at %s" % self.current_state)