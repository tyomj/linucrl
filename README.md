## LinUCRL for Recommender Systems

Based on [paper](http://papers.nips.cc/paper/7447-fighting-boredom-in-recommender-systems-with-linear-reinforcement-learning.pdf)
 "Fighting Boredom in Recommender Systems with Linear Reinforcement Learning", Romain WARLOP, Alessandro Lazaric, Jérémie Mary.


This repository contains:
 - an implementation of LinUCRL algorithm;
 - preprocessing functions for validation on MovieLens-1m dataset.

**Disclaimer**

This is an unofficial implementation and using this code you may not achieve the same results that were described in the paper ¯\_(ツ)_/¯.

### Requirements

Requires Python 3.6. Tested on Ubuntu 16.04 only.
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

All the experiments was initially run using [Ocean](https://github.com/olferuk/Ocean) framework for Data Science research.
