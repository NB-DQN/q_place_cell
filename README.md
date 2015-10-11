## About

The purpose is to computationally simulate the paper below, which showed that place fields in rat hippocampus are finner near the goal location in a maze task.

Hollup, S. A., Molden, S., Donnett, J. G., Moser, M. B., & Moser, E. I. (2001). Accumulation of hippocampal place fields at the goal location in an annular watermaze task. The Journal of Neuroscience, 21(5), 1635-1644.


## Method

* Maze environment

```
    0   1   2   3   4   5   6   7   8  
  8888888888888888888888888888888888888  
8 8                                 G 8 8
  8   88888888888888888888888888888   8  
7 8   8                           8   8 7
  8   8   8   8   8   8   8   8   8   8  
6 8   8                           8   8 6
  8   8   8   8   8   8   8   8   8   8  
5 8   8                           8   8 5
  8   8   8   8   8   8   8   8   8   8  
4 8   8                           8   8 4
  8   8   8   8   8   8   8   8   8   8  
3 8   8                           8   8 3
  8   8   8   8   8   8   8   8   8   8  
2 8   8                           8   8 2
  8   8   8   8   8   8   8   8   8   8  
1 8   8                           8   8 1
  8   88888888888888888888888888888   8  
0 8                                   8 0
  8888888888888888888888888888888888888
    0   1   2   3   4   5   6   7   8  

```

The starting point is always (0, 0) .

The goal is at first unavailable, but turns available once the agent has passed the entire arena. This trick solves the sampling problem that a well-trained agent would rarely visit all regions of the arena. 

* Agent

The agent chooses the direction in a Q-learning-based epsilon greedy manner. 

The agent gets some reward on the goal (indicated as G) if the goal is available, while  it gets no reward if the goal is unavailable.

Once the agent reaches the available goal, the environment is reset and the agent is put back to the starting point. 

* Training procedure

Training for predicting visual image following the movement is performed at the same time as the agent's Q learning.

* Validation and testing

Validation and testing are performed in every ``` valid_len ``` epochs. Dataset for validatioon and testing are generated randomly and independently.
