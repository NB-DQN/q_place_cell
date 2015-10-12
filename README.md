# About

The purpose is to computationally simulate the paper below, which showed that place fields in rat hippocampus are finner near the goal location in a maze task.

Hollup, S. A., Molden, S., Donnett, J. G., Moser, M. B., & Moser, E. I. (2001). Accumulation of hippocampal place fields at the goal location in an annular watermaze task. The Journal of Neuroscience, 21(5), 1635-1644.


# Methods

## Maze environment

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

The starting point is always (0, 0) . The goal is at first unavailable, but turns available once the agent has passed the entire arena. This trick solves the sampling problem that a well-trained agent would rarely visit all regions of the arena. 

## Agent

The agent chooses the direction in a Q-learning-based epsilon greedy manner. The agent gets some reward on the goal (indicated as G) if the goal is available, while it gets no reward if the goal is unavailable. Once the agent reaches the available goal, the environment is reset and the agent is put back to the starting point. 

## Training procedures

Two training procecure are independently implemented in thie repositry. The main training loop is LSTM, which learns to predict the visual image following the movement.

### A: train_LSTM.py

LSTM is trained using dataset generated by an agent in a maze, who chooses actions in a Q learning-based epsilon greedy manner. Validation and testing for LSTM are performed in every ``` valid_len ``` epochs with dataset generated randomly and independently. However, TRAINING DOES NOT WORK WELL, perhaps due to extremely biased training dataset.

### B: train_LSTM_with_pretrain.py

This training procedure is implemented in order to circumvent some problems on the previous training procedure.

* Accumulation of place fields near the goal locations is observed in adult rodents, while LSTM training in the previous learning procedure mimics rodents in developmental stages, since random weights are used for the initial parameters.
* Accumulation of place fields near the goal locations is observed in CA1 place cells, not in CA3 place cells.
* The previous training procedure does not work well.
    
1. pre-training of LSTM:
    This mimics rodents in developmental stages.
    * LSTM is trained using randomly generated training dataset.
    * In every ``` valid_len1 ``` epochs, weights on LSTM are clipped and:
        * LSTM validation is preformed
        * Using randomly generated dataset, SVM training (grid-search) and SVM testing are performed
        * When LSTM and SVM are trained enough, learning loop is broken
2. fine-tunign of LSTM:
    This mimics adult rodents undergoing annular water maze task. 
    * LSTM is trained using dataset generated by an agent in a maze.
    * The agent chooses actions in a Q learning-based epsilon greedy manner.
    * In every ``` valid_len2 ``` epochs, weights on LSTM are clipped and:
        * LSTM validation is performed
        * SVM testing is perfomed
        * When Q learning saturates, learning loop is broken
3. Clustering:
    EM clustering is perfomed on randomly generated dataset.

The RNN hdden layer and SVM targets are assumed in CA3 regions, while EM targets are assumed in CA1 regions where plastic place cells are observed.

# Results

## Training procedure A

## Training procedure B

```
[LSTM pre-train]
epoch 50: train perp: 725.21  valid square-sum error: 394.17 (3.76 epochs/sec)
SVM test accuracy: 1.0
epoch 100: train perp: 155.33  valid square-sum error: 37.54 (1.58 epochs/sec)
SVM test accuracy: 1.0
epoch 150: train perp: 40.80  valid square-sum error: 5.69 (1.92 epochs/sec)
SVM test accuracy: 1.0
epoch 200: train perp: 18.46  valid square-sum error: 2.47 (2.03 epochs/sec)
SVM test accuracy: 0.9996
epoch 250: train perp: 7.62  valid square-sum error: 0.83 (2.02 epochs/sec)
SVM test accuracy: 1.0
epoch 300: train perp: 2.68  valid square-sum error: 0.34 (2.04 epochs/sec)
SVM test accuracy: 1.0
epoch 350: train perp: 1.89  valid square-sum error: 0.31 (2.01 epochs/sec)
SVM test accuracy: 1.0
epoch 400: train perp: 1.66  valid square-sum error: 0.29 (1.87 epochs/sec)
SVM test accuracy: 1.0
epoch 450: train perp: 1.64  valid square-sum error: 0.30 (1.82 epochs/sec)
SVM test accuracy: 1.0
epoch 500: train perp: 1.21  valid square-sum error: 0.06 (1.59 epochs/sec)
SVM test accuracy: 1.0
epoch 550: train perp: 1.21  valid square-sum error: 0.05 (1.51 epochs/sec)
SVM test accuracy: 1.0
epoch 600: train perp: 1.17  valid square-sum error: 0.02 (1.67 epochs/sec)
SVM test accuracy: 1.0
epoch 650: train perp: 1.03  valid square-sum error: 0.02 (1.88 epochs/sec)
SVM test accuracy: 1.0
epoch 700: train perp: 0.69  valid square-sum error: 0.01 (1.98 epochs/sec)
SVM test accuracy: 1.0
epoch 750: train perp: 0.63  valid square-sum error: 0.01 (2.00 epochs/sec)
SVM test accuracy: 1.0
epoch 800: train perp: 0.63  valid square-sum error: 0.01 (1.83 epochs/sec)
SVM test accuracy: 1.0
epoch 850: train perp: 0.55  valid square-sum error: 0.01 (1.58 epochs/sec)
SVM test accuracy: 1.0
epoch 900: train perp: 0.50  valid square-sum error: 0.01 (1.67 epochs/sec)
SVM test accuracy: 1.0
[LSTM fine-tuning]
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

Q steps: 304
Q steps: 416
Q steps: 376
Q steps: 514
Q steps: 234
Q steps: 140
Q steps: 488
Q steps: 24
Q steps: 228
Q steps: 68
epoch 10: train perp: 3.89  valid square-sum error: 0.01 (0.39 epochs/sec)
SVM test accuracy: 1.0
Q steps: 334
Q steps: 46
Q steps: 82
Q steps: 74
Q steps: 42
Q steps: 90
Q steps: 148
Q steps: 32
Q steps: 162
Q steps: 46
epoch 20: train perp: 0.37  valid square-sum error: 0.01 (1.99 epochs/sec)
SVM test accuracy: 1.0
Q steps: 32
Q steps: 54
Q steps: 22
Q steps: 16
Q steps: 16
Q steps: 16
Q steps: 16
Q steps: 16
Q steps: 20
Q steps: 20
epoch 30: train perp: 0.13  valid square-sum error: 0.01 (3.35 epochs/sec)
SVM test accuracy: 1.0
Q steps: 18
Q steps: 18
Q steps: 20
Q steps: 16
Q steps: 18
Q steps: 18
Q steps: 18
Q steps: 16
Q steps: 16
Q steps: 16
epoch 40: train perp: 0.11  valid square-sum error: 0.01 (3.02 epochs/sec)
SVM test accuracy: 1.0
[Clustering]
[(0, 1), (0, 16), (1, 21), (1, 27), (2, 8), (2, 24), (3, 2), (3, 13), (4, 2), (4, 22), (5, 9), (5, 20), (6, 9), (6, 31), (7, 4), (7, 31), (8, 4), (8, 10), (9, 10), (9, 28), (10, 10), (10, 28), (11, 10), (11, 19), (12, 19), (13, 19), (13, 29), (14, 19), (14, 29), (15, 29), (16, 17), (16, 29), (17, 17), (18, 17), (19, 17), (20, 0), (20, 17), (21, 0), (21, 17), (22, 0), (22, 17), (23, 11), (23, 15), (24, 11), (24, 30), (25, 15), (25, 18), (26, 5), (26, 15), (27, 7), (27, 23), (28, 7), (28, 14), (29, 25), (29, 26), (30, 3), (30, 6), (31, 3), (31, 12)]
[test]
test square-sum error: 0.00

```
