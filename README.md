## LinUCRL for Recommender Systems

Based on [paper](http://papers.nips.cc/paper/7447-fighting-boredom-in-recommender-systems-with-linear-reinforcement-learning.pdf)
 "Fighting Boredom in Recommender Systems with Linear Reinforcement Learning", Romain WARLOP, Alessandro Lazaric, Jérémie Mary.


This repository contains:
 - an implementation of LinUCRL algorithm;
 - preprocessing functions for running on MovieLens-1m dataset.

### Requirements

Requires Python 3.6 and tested on Ubuntu 16.04.
Please check out `requirements.txt` for resolving dependency issues.

### Quick start

1. Clone this repository
2. Download MovieLens dataset
    ```
    make -B download_dataset
    ```
3. Run an experiment
    ```
    make -B train
    ```

### Configuration

Check out configuration file `lucrl/config/config.yaml` which contains parameters for dataset, mdp and LinUCRL algorithm.

### Acknowledgments

1. All the experiments was initially run using [Ocean](https://github.com/olferuk/Ocean) framework for Data Science research.
2. Thanks to Romain for providing me with [original](https://github.com/RomainWarlop/ReinforcementLearning/tree/master/ReinforcementLearning) implementation when I got stuck.