import argparse
import math
import sys
import time
import random
import pickle

import numpy as np
import six
import matplotlib.pyplot as plt

import chainer
from chainer import cuda
import chainer.functions as F
from chainer import optimizers

from dataset_generator import DatasetGenerator
import environment

import q_agent

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from sklearn.mixture import GMM
from sklearn import metrics

# LSTM parameters
# n_epoch = 1000 # number of epochs
n_units = 60 # number of units per layer
batchsize = 1 # minibatch size
bprop_len = 1 # length of truncated BPTT
valid_len1 = 50 # epoch on which accuracy and perp are calculated: pre-training phase
valid_len2 = 10 # epoch on which accuracy and perp are calculated: fine-tuning phase
grad_clip = 5 # gradient norm threshold to clip
train_len = 100 # length of training dataset on pre-training phase

# Environment parameters
maze_size = (9, 9)
goal_location = (9, 5)

# SVM and clustering parameters
ev_iterations = 100 # iterations for generating SVM and clustering dataset

# GPU
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int,
        help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()
mod = cuda.cupy if args.gpu >= 0 else np

# LSTM validation dataset: random
valid_data = DatasetGenerator(maze_size).generate_seq_random(100)

# LSTM model
model = chainer.FunctionSet(
        x_to_h = F.Linear(64, n_units * 4),
        h_to_h = F.Linear(n_units, n_units * 4),
        h_to_y = F.Linear(n_units, 60))
if args.gpu >= 0:
    cuda.check_cuda_available()
    cuda.get_device(args.gpu).use()
    model.to_gpu()

# LSTM optimizer
optimizer = optimizers.SGD(lr=1.)
optimizer.setup(model.collect_parameters())

# LSTM one-step forward propagation
def forward_one_step(x, t, state, train=True):
    # if args.gpu >= 0:
    #     data = cuda.to_gpu(data)
    #     targets = cuda.to_gpu(targets)
    x = chainer.Variable(x, volatile=not train)
    t = chainer.Variable(t, volatile=not train)
    h_in = model.x_to_h(x) + model.h_to_h(state['h'])
    c, h = F.lstm(state['c'], h_in)
    y = model.h_to_y(h)
    state = {'c': c, 'h': h}

    sigmoid_y = 1 / (1 + np.exp(-y.data))
    bin_y = np.round((np.sign(sigmoid_y - 0.5) + 1) / 2)

    square_sum_error = ((t.data - sigmoid_y) ** 2).sum()
    bin_y_error = ((t.data - bin_y) ** 2).sum()
    
    return state, F.sigmoid_cross_entropy(y, t), square_sum_error, bin_y_error, h.data[0]
    
# LSTM initializing hidden state
def make_initial_state(batchsize=batchsize, train=True):
    return {name: chainer.Variable(mod.zeros((batchsize, n_units),
        dtype=np.float32),
        volatile=not train)
        for name in ('c', 'h')}

# LSTM evaluation
def evaluate(data, test=False):
    sum_error = 0.0
    state = make_initial_state(batchsize=1, train=False)
    hh = []
    bin_y_error_sum = 0.0
    
    for i in six.moves.range(len(data['input'])):
        x_batch = mod.asarray([data['input'][i]], dtype = 'float32')
        t_batch = mod.asarray([data['output'][i]], dtype = 'int32')
        state, loss, square_sum_error, bin_y_error, h_raw = forward_one_step(x_batch, t_batch, state, train=False)
        
        hh.append(h_raw)
        bin_y_error_sum += bin_y_error
        sum_error += square_sum_error
        
    return sum_error, hh, bin_y_error_sum

# SVM and EM: generate dataset
def generate_seq_sklearn(iterations):
    label = []
    input_data = []
    for i in range(iterations):
        test_data = DatasetGenerator(maze_size).generate_seq_random(100)
        test_square_sum_error, test_hh, test_error = evaluate(test_data, test=True)
        label.extend(test_data['coordinates'])
        input_data.extend(test_hh)
        
    return input_data , label

# Clustering function
def EM_clustering():    
    # generate dataset for clustering
    clustering_input_data, clustering_label  = generate_seq_sklearn(ev_iterations)
    
    # data allocation
    clst_X_train, clst_X_test, clst_y_train, clst_y_test = train_test_split(clustering_input_data, clustering_label)
    
    # clustering: EM
    cid_length = (maze_size[0] + maze_size[1] - 2) * 2 # 32
    target_length = cid_length / 2 # 16
    EM_classifier = GMM(n_components= target_length, n_iter=500, covariance_type='tied')
    EM_classifier.fit(clst_X_train)
    clst_y_train_pred = EM_classifier.predict(clst_X_train) 
    
    # show results of EM clustering
    print('')
    print('[Clustering]')
    labels_list = zip(clst_y_train, clst_y_train_pred)
    labels_list.sort()
    # print(labels_list)
    # print unique label set
    labels_list_unique = []
    for x in labels_list:
        if x not in labels_list_unique:
            labels_list_unique.append(x)
    print('Clustering results: (coordinate ID, cluster ID)')
    print(labels_list_unique)
    
    # analysis: size of place fields
    result_size = np.zeros((target_length, 3))
    for cluster_id in range(target_length):
        cid_list = []
        for i in range(len(labels_list_unique)):
            if labels_list_unique[i][1] == cluster_id:
                cid_list.append(labels_list_unique[i][0])
        
        mean_cid = float(sum(cid_list)) / len(cid_list)
        distribution_cid = max(cid_list) - min(cid_list) + 1
        result_size[cluster_id] = [cluster_id, mean_cid, distribution_cid]
    result_size_cut = np.delete(result_size, 0, 1) # [mean_cid, distribution_cid]
    result_size_sorted = result_size_cut[result_size_cut[:, 0].argsort()] # sort by mean_cid
    
    # analysis: distribution of place fields
    result_distribution = np.zeros((cid_length, 2))
    for cid in range(cid_length):
        cluster_count = 0
        for i in range(len(labels_list_unique)):
            if labels_list_unique[i][0] == cid:
                cluster_count += 1
        result_distribution[cid] = [cid, cluster_count]
    return EM_classifier, result_size_sorted, result_distribution


# loop initialization
cur_log_perp = mod.zeros(())
start_at = time.time()
cur_at = start_at
epoch = 0
accum_loss = chainer.Variable(mod.zeros((), dtype=np.float32))

# Pretraining LSTM: train LSTM with random dataset
print('')
print('[LSTM pre-training]')
while True:
    
    # LSTM
    # initialize hidden state to 0
    state = make_initial_state()

    # generate LSTM training dataset
    train_data = DatasetGenerator(maze_size).generate_seq_random(train_len)
    
    for i in six.moves.range(train_len):

        # LSTM training dataset
        x_batch = mod.array([train_data['input'][i]],  dtype = 'float32')
        t_batch = mod.array([train_data['output'][i]], dtype = 'int32')
        
        # LSTM one step forward propagation
        state, loss_i, acc_i, bin_i, h_i = forward_one_step(x_batch, t_batch, state)
        accum_loss += loss_i
        cur_log_perp += loss_i.data.reshape(())

        # LSTM truncated BPTT
        optimizer.zero_grads()
        accum_loss.backward()
        accum_loss.unchain_backward()  # truncate
        accum_loss = chainer.Variable(mod.zeros((), dtype=np.float32))
        optimizer.clip_grads(grad_clip) # gradient clip
        optimizer.update()

        sys.stdout.flush()

    if (epoch + 1) % valid_len1 == 0:
    
        # calculate LSTM accuracy, cumulative loss & throughput
        valid_square_sum_error, valid_hh, valid_error = evaluate(valid_data)
        perp = cuda.to_cpu(cur_log_perp) / valid_len1
        now = time.time()
        throughput = valid_len1 / (now - cur_at)
        print('epoch {}: train perp: {:.2f}  valid square-sum error: {:.2f} ({:.2f} epochs/sec)'
                .format(epoch+1, perp, valid_square_sum_error, throughput))
        cur_at = now
        
        # SVM
        # generate dataset for SVM
        SVM_input_data, SVM_output_data  = generate_seq_sklearn(ev_iterations)
        
        # SVM data allocation
        SVM_X_train, SVM_X_test, SVM_y_train, SVM_y_test = train_test_split(SVM_input_data, SVM_output_data)

        # SVM grid search parameters
        tuned_parameters = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
        
        # SVM grid search
        clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, scoring='accuracy')
        clf.fit(SVM_X_train, SVM_y_train)
        
        # SVM test
        y_true, y_pred = SVM_y_test, clf.predict(SVM_X_test)
        SVM_test_accuracy = accuracy_score(y_true, y_pred)
        print('SVM test accuracy: {}'.format(SVM_test_accuracy))
        
        # termination criteria
        if perp < 0.5 and valid_square_sum_error < 0.01 and SVM_test_accuracy > 0.99:
            break
        else:
            cur_log_perp.fill(0)
            
    epoch += 1

# Clustering
EM_classifier1, result_size_before, result_distribution_before = EM_clustering()

# Fine tuning: train LSTM with Q-learning based dataset
epoch = 0
print('')
print('[LSTM fine-tuning]')

# Q environment and agent
env =environment.Environment(maze_size, goal_location)
env.maze.display_cui()
agent = q_agent.QAgent(env)
count_move_mean = 0

while True:

    # initialize hidden state to 0
    state = make_initial_state()
    
    # record the agent's movement
    count_move = 0
    direction_history = []
    cid_history = []

    while not env.get_goal():
        # choose action based on Q learning
        cid, direction_int, next_cid = agent.choose_action()
        direction = [0] * 4
        direction[direction_int] = 1 # one-hot
        
        # get virtual image
        image = env.visual_image(cid)
        next_image = env.visual_image(next_cid)
        
        # LSTM training dataset
        x_batch = mod.array([direction + image.tolist()],  dtype = 'float32')
        t_batch = mod.array([next_image], dtype = 'int32')
        
        # LSTM one step forward propagation
        state, loss_i, acc_i, bin_i, h_i = forward_one_step(x_batch, t_batch, state)
        accum_loss += loss_i
        cur_log_perp += loss_i.data.reshape(())
        
        # LSTM truncated BPTT
        optimizer.zero_grads()
        accum_loss.backward()
        accum_loss.unchain_backward()  # truncate
        accum_loss = chainer.Variable(mod.zeros((), dtype=np.float32))
        optimizer.clip_grads(grad_clip) # gradient clip
        optimizer.update()
        
        # Q history update
        count_move += 1
        direction_history.append(direction_int)
        cid_history.append(cid)
        
        sys.stdout.flush()
        
    env.reset()
    print('Q steps: {}'.format(count_move))
    count_move_mean += count_move

    if (epoch + 1) % valid_len2 == 0:
        
        # print the movement histories
        # print('direction history: {}'.format(direction_history))
        # print('cid history: {}'.format(cid_history))
    
        # calculate LSTM accuracy, cumulative loss & throuput
        valid_square_sum_error, valid_hh, valid_error = evaluate(valid_data)
        perp = cuda.to_cpu(cur_log_perp) / valid_len2
        now = time.time()
        throuput = valid_len2 / (now - cur_at)
        print('epoch {}: train perp: {:.2f}  valid square-sum error: {:.2f} ({:.2f} epochs/sec)'
                .format(epoch+1, perp, valid_square_sum_error, throuput))
        cur_at = now
        
        # SVM test to check that test error of SVM is 0
        # generate dataset for SVM test
        SVM_input_data, SVM_output_data  = generate_seq_sklearn(ev_iterations/5)
        
        # SVM test
        y_true, y_pred = SVM_output_data, clf.predict(SVM_input_data)
        SVM_test_accuracy = accuracy_score(y_true, y_pred)
        print('SVM test accuracy: {}'.format(SVM_test_accuracy))
        
        # termination criteria
        count_move_mean = count_move_mean / valid_len2
        
        if count_move_mean < 19:
            break
        else:
            cur_log_perp.fill(0)
            count_move_mean = 0
            
    epoch += 1
            
# Clustering
EM_classifier2, result_size_after, result_distribution_after = EM_clustering()

# save the LSTM model
f = open('LSTM_model_' + str(maze_size[0]) + '_' + str(maze_size[1]) + '.pkl', 'wb')
pickle.dump(model, f, 2)
f.close()

# save the SVM model
f = open('SVM_model_'  + str(maze_size[0]) + '_' + str(maze_size[1]) + '.pkl', 'wb')
pickle.dump(clf, f, 2)
f.close()  

# save the EM model
f = open('EM_model_'  + str(maze_size[0]) + '_' + str(maze_size[1]) + '.pkl', 'wb')
pickle.dump(EM_classifier2, f, 2)
f.close()  

# LSTM evaluate on test dataset
print('')
print('[LSTM test]')
test_data = DatasetGenerator(maze_size).generate_seq_random(100)
test_square_sum_error, test_hh, test_bin_y_error = evaluate(test_data, test=True)
print('test square-sum error: {:.2f}'.format(test_square_sum_error))

# plot the clustering results
# size analysis
plt.subplot(1, 2, 1)
plt. plot(result_size_before[:, 0], result_size_before[:, 1], c='blue', marker="o")
plt.hold(True)
plt. plot(result_size_after[:, 0], result_size_after[:, 1], c = 'red', marker="o")
plt.title("Size of place field for each place cell cluster center")
plt.xlabel("Coordinate ID")
plt.ylabel("Size of place field")
plt.legend(["Before Training", "After Training"], loc =3)

# distribution analysis
plt.subplot(1, 2, 2)
plt. plot(result_distribution_before[:, 0], result_distribution_before[:, 1], c='blue', marker="o")
plt.hold(True)
plt.plot(result_distribution_after[:, 0], result_distribution_after[:, 1], c = 'red', marker="o")
plt.title("Distribution of place cells for each coordinates")
plt.xlabel("Coordinate ID")
plt.ylabel("Number of place cells")
plt.legend(["Before Training", "After Training"], loc =3)
plt.show()
