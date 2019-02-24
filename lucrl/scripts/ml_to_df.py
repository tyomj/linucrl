import os
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from joblib import Parallel, delayed

project_dir = os.path.normpath(os.path.dirname(os.path.abspath(__file__)) + os.sep + os.pardir)

from lucrl.utils.coordinator import Coordinator
crd = Coordinator(project_dir)

from lucrl.utils.logger import Logger
logger = Logger(project_dir, 'ml_to_df', 'ml_to_df.txt')

with open(os.path.join(crd.config.path, 'config.yaml')) as f:
    config = yaml.load(f)
WORKERS_COUNT = config.get('system_workers_count', 1)


def genres_label_encoding(df: pd.DataFrame):
    """Based on genres column which by default contains string of genres separated by `|` creates two columns:
    - `genres_name` which contains only first genre as a main genre
    - `genres` where instead of names we encode out genres as indexes from the original list of MovieLens genres, defined in the config file.
    """

    df.genres = df.genres.str.replace("Children's", "Children")

    def _get_one_genre(entry):
        movie_genres = entry.split("|")
        output = None
        for i, genre in enumerate(config['dataset']['GENRES']):
            if genre in movie_genres[:1]:
                output = genre
        return output

    def _get_index_of_genre(entry):
        for i, genre in enumerate(config['dataset']['GENRES']):
            if genre == entry:
                output = i
        return output

    df['genres_name'] = df['genres'].apply(_get_one_genre)
    df['genres'] = df['genres_name'].apply(_get_index_of_genre)

    return df

def load_and_merge():
    """
    Loads data from path defined in config file
    :return: pd.DataFrame
    """
    movies = pd.read_csv(os.path.join(crd.data_raw.path, config['dataset']['movies_path']))
    ratings = pd.read_csv(os.path.join(crd.data_raw.path, config['dataset']['ratings_path']))
    movies = genres_label_encoding(movies)
    ratings = ratings.merge(movies[['item_id', 'genres']]).reset_index(drop=True)
    ratings = ratings.astype(int)
    return ratings

def filter_genres(df: pd.DataFrame, min_occurrence=10000):
    """
    Filters rare genres.

    :param df: dataframe with genres columns
    :param min_occurrence: min number of rated reviews per genre
    :return: modified DataFrame
    """
    df = df[
        df.genres.isin(df.genres.value_counts()[df.genres.value_counts() > min_occurrence].index)].reset_index(
        drop=True)
    return df

def get_list(series):
    """
    Series to list.

    :param series:
    :return:
    """
    return series.reset_index(drop=True).tolist()


def construct_states(group, window_size=5):
    """
    Is used for constructing a state from a sequence of user's history with the len == `window_size`.

    :param group: an instance of pd.core.groupby object
    :param window_size: number of previous action to limit the session history
    :return: modified group
    """
    series_for_state = group.genres
    series_for_reward = group.rating
    s_states = [None for _ in range(len(series_for_state))]
    s_rewards = [None for _ in range(len(series_for_state))]
    for i in range(len(series_for_state) - (window_size - 1)):
        state_slice = series_for_state[i:i + window_size]
        reward_slice = series_for_reward[i:i + window_size]
        if state_slice.isnull().sum() == 0:
            state = get_list(state_slice)
            reward = reward_slice.values
            s_states[i + (window_size - 1)] = state
            s_rewards[i + (window_size - 1)] = reward
    group['state'] = s_states
    group['initial_reward'] = s_rewards

    return group

def mean_init_reward_in_group(a):
    output = [np.stack(a.values).mean(axis=0)]
    return output

def get_ml_df():
    """
    A wrapper for the whole preparation process to obtain a DataFrame with the states.

    :return:
    """

    df = load_and_merge()
    df = filter_genres(df, config['dataset']['min_num_of_ratings'])
    df = df.sort_values(['user_id', 'timestamp'], ascending=True).reset_index(drop=True)

    result = Parallel(n_jobs=WORKERS_COUNT)(delayed(construct_states)(group_, config['mdp']['window_size']) \
                                 for name_, group_ in tqdm(df.groupby(['user_id'])))
    df = pd.concat(result)
    # get rid of states without full window size history
    df = df[~df.state.isnull()].reset_index(drop=True)

    df['state_str'] = df.state.apply(lambda x: '-'.join([str(x1) for x1 in x]))
    df = df[['rating', 'state_str', 'genres','initial_reward']]

    # for faster convergence we will use mean value instead of randomized sample from a specific group.
    df = df.groupby('state_str').agg(
        {'rating': 'mean', 'genres': 'mean', 'initial_reward': mean_init_reward_in_group}).reset_index()
    df = df.rename({'state_str': 'state', 'rating': 'reward'}, axis=1)
    df.genres = df.genres.astype(int)

    logger.info("Window size: {}".format(config['mdp']['window_size']))
    logger.info("Number of unique states: {}".format(df.state.nunique()))
    logger.info("Number of unique actions: {}".format(df.genres.nunique()))

    return df


