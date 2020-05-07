# Homer Algorithm 

Code for Homer Algorithm developed at Microsoft Research, New York.

Please don't share it outside Microsoft.

# Installation

Make sure you are using Python 3.6, 3.5 which is the default on the GPU cluster gives Pytorch multiprocessing errors.

First install some dependencies for Atari:

`sudo pip install gym[atari]`

# How to run an experiment

`export PYTHONPATH=$PYTHONPATH:src`

`python3 src/experiments/run_homer.py --horizon 10 --env diabcombolock --noise hadamhardg --name first-run`
