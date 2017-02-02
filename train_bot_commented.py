import datetime
import os
import sys

import numpy as np
import json

from keras.models import Sequential,load_model
from keras.layers import Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard
from keras.optimizers import SGD, RMSprop

# credit to https://github.com/brianvanleeuwen/Halite-ML-starter-bot for the starter code

# There are many different optimizer formulas that you can use from Keras
# Each of them are some form of gradient decent algorithm
sgd = SGD(lr=0.01, momentum=0.75, decay=1e-8, nesterov=False)

# With optimizers come several parameters that we can change, I'll explain what a couple of those are.
#  lr: this is our learning rate (called it steps in class) they are similar if not the same thing
#       with a high learning rate we can potentially pass our convergence point at the minima and head in the wrong dir
#       with a too low learning rate we take way too long to get to our convergence point
#       trial and error seems to work the best, and some problems have optimized parameters for them already
#  momentum: https://en.wikipedia.org/wiki/Gradient_descent#The_momentum_method
#       wikipedia has a concise explanation of momentum - about two sentences
#       more mathematical - http://ufldl.stanford.edu/tutorial/supervised/OptimizationStochasticGradientDescent/
#  decay: basically how much we want our weights to change by as we make steps along the curve
#       http://stats.stackexchange.com/questions/29130/difference-between-neural-net-weight-decay-and-learning-rate
#  nesterov: further optimization - from what I understand is you might step a bit farther than what it normall would

#  good resource on optimizers - http://sebastianruder.com/optimizing-gradient-descent/

REPLAY_FOLDER = sys.argv[1]
training_input = []
training_target = []

# This variable and the same one in MyBot.py must be the same
# we can change this - either increase/decrease to get a different model
VISIBLE_DISTANCE = 4  # how many squares out we will be looking
input_dim=4*(2*VISIBLE_DISTANCE+1)*(2*VISIBLE_DISTANCE+1)
# input dimensions - with the base value of 4 explanation
# we are looking in 4 directions and multiply that by 9 and by 9 again
# essentially, we have a 9x9 square that is centered on each square that we are looking at

np.random.seed(0) # for reproducibility

# This the basis for how we are going to both form and train our model
# we are using three convolutions neural networks https://en.wikipedia.org/wiki/Convolutional_neural_network
# this is also - in my opinion - one of the more complex aspects of this code
#  here's a good resource on information regarding neural networks http://cs231n.github.io/neural-networks-1/

# another resource - http://cs231n.github.io/convolutional-networks/
model = Sequential([Dense(512, input_dim=input_dim),
                    LeakyReLU(),
                    Dense(512),
                    LeakyReLU(),
                    Dense(512),
                    LeakyReLU(),
                    Dense(5, activation='softmax')])

# This is how we are finding our 'line of best fit' to classify our data so that we can make predictions
# keras offers different optimizers - while they are similar some are better suited for different tasks
# I have found that rmsprop works pretty well for this problem
model.compile('rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

def stack_to_input(stack, position):
    return np.take(np.take(stack,
                np.arange(-VISIBLE_DISTANCE,VISIBLE_DISTANCE + 1)+position[0],axis=1,mode='wrap'),
                np.arange(-VISIBLE_DISTANCE,VISIBLE_DISTANCE + 1)+position[1],axis=2,mode='wrap').flatten()

size = len(os.listdir(REPLAY_FOLDER))   # how many replays that we are going to load in and train on
for index, replay_name in enumerate(os.listdir(REPLAY_FOLDER)):
    #  parsing the replay files using the JSON package
    if replay_name[-4:]!='.hlt':continue  # only grab .hlt files
    print('Loading {} ({}/{})'.format(replay_name, index, size))  # command formatting to look pretty
    replay = json.load(open('{}/{}'.format(REPLAY_FOLDER,replay_name))) # load the replay from reading the JSON format

    #  What if we want to target a specific bot?? Are there any advantages?
    #  One of the most important aspects of machine learning is to not train your model saying that left=right
    #  it will most likely do nothing.

    #  Going off of this, let's not train off of the winning player whoever that may be
    #  we are going to train on 'erdman v17' - in this example

    player_array = replay['player_names']
    target_id = player_array.index('erdman v17')


    #  There are however some implications of this line that we have to account for with more tedious methods.

    #  First we need to make sure that erdman v17 is in the replay that we are watching
    #  Second, because we are no longer skipping when a neutral tile is the largest at the end we run into an issue.
    #       Sometimes the servers will crap out and nobody will be able to move at all - we don't check for this
    #  To rectify this situation we have to pre-screen the replays by sorting them by size and deleting the ones
    #       that are much too small - make sure not to delete all the small ones because they might be actual games
    #       I found that anything under 50KB was usually a dead game



    # https://halite.io/advanced_replay_file.php
    frames=np.array(replay['frames'])
    player=frames[:,:,:,0]

    #  WINNER determiner
    #  what we are doing here is looking at the 4th Dimension of the Frame array from the replay files
    #  players will store all the player ids (0 = unowned, 1+ are all actual players) and the strength of their cells
    #  We look at the largest counts (cell strength) and assume that that is the winner
    #  If the player who owns it is 0, that means it is unowned and we should skip the game

    #  The issue with this idea of training on only the winner is that we might train our bot that
    #  going left is good and bad and going right is good and bad due to different strategies.
    #  This would result in the bot not really doing anything to minimize that loss function
    ######################################################################
    #  We don't need this code anymore to determine the winner
    # players,counts = np.unique(player[-1],return_counts=True)
    # target_id = players[counts.argmax()]
    # if target_id == 0: continue
    ######################################################################

    # get all the productions
    prod = np.repeat(np.array(replay['productions'])[np.newaxis],replay['num_frames'],axis=0)
    strength = frames[:,:,:,1]  # set the strengths

    # getting the moves that our target player/bot is making and only caring about the first ~128 moves or so
    # if we wanted we could use this mentality to craft early and late game models
    moves = (np.arange(5) == np.array(replay['moves'])[:,:,:,None]).astype(int)[:128]
    #
    stacks = np.array([player==target_id,(player!=target_id) & (player!=0),prod/20,strength/255])
    # numpy transposition - https://docs.scipy.org/doc/numpy/reference/generated/numpy.transpose.html
    # we are dividing the production by 20 and strength by 255 for regularization - basically limit out noise
    #       so that we try to avoid overfitting our model
    stacks = stacks.transpose(1,0,2,3)[:len(moves)].astype(np.float32)

    position_indices = stacks[:,0].nonzero()

    # we take samples from our games
    # for example I can load in 500 replay files and will then get around 600K samples to train on
    sampling_rate = 1/stacks[:,0].mean(axis=(1,2))[position_indices[0]]

    #  1 is standing still, we want to favor moving more than staying still, so we put a higher default weight
    sampling_rate *= moves[position_indices].dot(np.array([1,10,10,10,10]))
    sampling_rate /= sampling_rate.sum()
    sample_indices = np.transpose(position_indices)[np.random.choice(np.arange(len(sampling_rate)),
                                                                    min(len(sampling_rate),2048),p=sampling_rate,replace=False)]

    replay_input = np.array([stack_to_input(stacks[i],[j,k]) for i,j,k in sample_indices])
    replay_target = moves[tuple(sample_indices.T)]
    
    training_input.append(replay_input.astype(np.float32))
    training_target.append(replay_target.astype(np.float32))

now = datetime.datetime.now()
tensorboard = TensorBoard(log_dir='./logs/'+now.strftime('%Y.%m.%d %H.%M'))
training_input = np.concatenate(training_input,axis=0)
training_target = np.concatenate(training_target,axis=0)
indices = np.arange(len(training_input))
np.random.shuffle(indices) #  shuffle our training samples to help avoid overfitting
training_input = training_input[indices]
training_target = training_target[indices]

# this is where we fit our model
# couple of key points, we cross validate on 20% of our data that we put into memory - avoids overfitting
# if we don't improve in 10 epochs we give up and stop
# max num of epochs is 1000
model.fit(training_input,training_target,validation_split=0.2,
          callbacks=[EarlyStopping(patience=10),
                     ModelCheckpoint('model.h5',verbose=1,save_best_only=True),
                     tensorboard],
          batch_size=1024, nb_epoch=1000)

model = load_model('model.h5')

still_mask = training_target[:,0].astype(bool)
print('STILL accuracy:',model.evaluate(training_input[still_mask],training_target[still_mask],verbose=0)[1])
print('MOVE accuracy:',model.evaluate(training_input[~still_mask],training_target[~still_mask],verbose=0)[1])
