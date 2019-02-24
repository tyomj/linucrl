import os
import sys
upper_project_dir = os.path.normpath(os.path.dirname(os.path.abspath(__file__)) + os.sep + os.pardir + os.sep + os.pardir)
sys.path.append(upper_project_dir)

from lucrl.scripts.ml_to_df import get_ml_df
from lucrl.scripts.mdp import MDP, check_mdp
from lucrl.scripts.linucrl import LinUCRL

def train():
    # create a DataFrame for mdp
    df_for_mdp = get_ml_df()

    # get MDB object
    mdp = MDP(df_for_mdp)

    # and check it
    check_mdp(mdp)

    # Fit LinUCRL
    lucrl = LinUCRL(mdp)
    lucrl.fit()


if __name__ == '__main__':
    train()
