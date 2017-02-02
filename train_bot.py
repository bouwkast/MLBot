import datetime
import os
import sys

import numpy as np
import json

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import SGD, RMSprop

# credit to https://github.com/brianvanleeuwen/Halite-ML-starter-bot for the starter code

# There are many different optimizer formulas that you can use from Keras
# Each of them are some form of gradient decent algorithm
sgd = SGD(lr=0.01, momentum=0.75, decay=1e-8, nesterov=False)

# several optimizers were used/tested and here are some findings:

# Stochastic Gradient Descent seemed to be the simplest algorithm, but because of that
#       it took up to 10 times longer to train (ex 500 games was around an hour w/ low learning rate)
#       Also, SGD got stuck in local minima about 50% of the time - ruining training

# Nadam seemed to converge relatively quickly, but compared to RMSProp it seemed to always be
#       slightly worse and from gathering info online it seems that momentum can overshoot your minima
#       This also tended to get stuck in local minima

# RMSProp gave us the best result - it was much quicker than SGD (as was Nadam) but we seemed to
#       get a higher move (N,E,S,W) accuracy by a few % points than the others - this is what our model used.

#  good resource on optimizers that was used - http://sebastianruder.com/optimizing-gradient-descent/

REPLAY_FOLDER = sys.argv[1]
training_input = []
training_target = []

# This variable and the same one in MyBot.py must be the same
# we can change this - either increase/decrease to get a different model
VISIBLE_DISTANCE = 4  # how many squares out we will be looking
input_dim = 4 * (2 * VISIBLE_DISTANCE + 1) * (2 * VISIBLE_DISTANCE + 1)
# we tried different distances - but the results were relatively indistinguishable
# we are looking in 4 directions and multiply that by 9 and by 9 again

np.random.seed(0)  # for reproducibility

# This the basis for how we are going to both form and train our model
# We are using three convolutions neural networks
# different sizes of the layers were tested - if you go too big (maybe twice this size) the model file
#       isn't accepted by Halite because of size restrictions

model = Sequential([Dense(512, input_dim=input_dim),
                    LeakyReLU(),
                    Dense(512),
                    LeakyReLU(),
                    Dense(512),
                    LeakyReLU(),
                    Dense(5, activation='softmax')])

# keras offers different optimizers - while they are similar some are better suited for different tasks
# rmsprop worked pretty well for this problem
model.compile('rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


def stack_to_input(stack, position):
    return np.take(np.take(stack,
                           np.arange(-VISIBLE_DISTANCE, VISIBLE_DISTANCE + 1) + position[0], axis=1, mode='wrap'),
                   np.arange(-VISIBLE_DISTANCE, VISIBLE_DISTANCE + 1) + position[1], axis=2, mode='wrap').flatten()


size = len(os.listdir(REPLAY_FOLDER))  # how many replays that we are going to load in and train on
for index, replay_name in enumerate(os.listdir(REPLAY_FOLDER)):
    #  parsing the replay files using the JSON package
    if replay_name[-4:] != '.hlt': continue  # only grab .hlt files
    print('Loading {} ({}/{})'.format(replay_name, index, size))  # command formatting to look pretty
    replay = json.load(open('{}/{}'.format(REPLAY_FOLDER, replay_name)))  # load the replay from reading the JSON format
    #  Originally, the bot trained on whoever had the highest strength piece for each game
    #  This caused some problems - namely that each person had a different strategy and
    #       ended up causing the bot to have conflicting data that it was training on

    #  Emulating a specific player for every game (don't care whether win/lose) allowed us to
    #       mimic their strategy much better

    #  Our current model was a combination of around 1500 of nmalaguti's rank 1 or 2 bots

    #  Player IDs start at 1, not 0
    player_array = replay['player_names']
    # target_id = player_array.index('erdman v17') + 1
    # target_id = player_array.index('nmalaguti v52') + 1
    # target_id = player_array.index('nmalaguti v53') + 1
    # target_id = player_array.index('nmalaguti v54') + 1
    # target_id = player_array.index('mzotkiew v23')
    if 'nmalaguti v54' in player_array:
        target_id = player_array.index('nmalaguti v54') + 1
    else:
        target_id = player_array.index('nmalaguti v53') + 1

    #  All replays were guaranteed to have one of those bots in them
    #  we also had to make sure to remove the early games b/c they played against bad players
    #  we also had to remove games where the server errored out.

    frames = np.array(replay['frames'])
    player = frames[:, :, :, 0]

    #  We don't need this code anymore to determine the "winner" only look at specific bots
    # players,counts = np.unique(player[-1],return_counts=True)
    # target_id = players[counts.argmax()]
    # if target_id == 0: continue


    # get all the productions
    prod = np.repeat(np.array(replay['productions'])[np.newaxis], replay['num_frames'], axis=0)
    strength = frames[:, :, :, 1]  # set the strengths

    # getting the moves that our target player/bot is making
    # we could truncate this array to craft early and late game models - which may have worked better
    moves = (np.arange(5) == np.array(replay['moves'])[:, :, :, None]).astype(int)
    #
    stacks = np.array([player == target_id, (player != target_id) & (player != 0), prod / 20, strength / 255])
    # we are dividing the production by 20 and strength by 255 for regularization - basically limit out noise
    #       so that we try to avoid overfitting our model
    stacks = stacks.transpose(1, 0, 2, 3)[:len(moves)].astype(np.float32)

    position_indices = stacks[:, 0].nonzero()

    # we take samples from our games
    # for example I can load in 500 replay files and will then get around 600K samples to train on
    sampling_rate = 1 / stacks[:, 0].mean(axis=(1, 2))[position_indices[0]]

    #  1 is standing still, we want to favor moving more than staying still, so we put a higher default weight
    sampling_rate *= moves[position_indices].dot(np.array([1, 10, 10, 10, 10]))
    sampling_rate /= sampling_rate.sum()
    sample_indices = np.transpose(position_indices)[np.random.choice(np.arange(len(sampling_rate)),
                                                                     min(len(sampling_rate), 2048), p=sampling_rate,
                                                                     replace=False)]

    replay_input = np.array([stack_to_input(stacks[i], [j, k]) for i, j, k in sample_indices])
    replay_target = moves[tuple(sample_indices.T)]

    training_input.append(replay_input.astype(np.float32))
    training_target.append(replay_target.astype(np.float32))

now = datetime.datetime.now()
tensorboard = TensorBoard(log_dir='./logs/' + now.strftime('%Y.%m.%d %H.%M'))
training_input = np.concatenate(training_input, axis=0)
training_target = np.concatenate(training_target, axis=0)
indices = np.arange(len(training_input))
np.random.shuffle(indices)  # shuffle our training samples to help avoid overfitting
training_input = training_input[indices]
training_target = training_target[indices]

# this is where we fit our model
# couple of key points, we cross validate on 20% of our data that we put into memory - avoids overfitting
# the amount to cross validate on was difficult to tune in hopefully 20% is good
# if we don't improve in 10 epochs we give up and stop
# max num of epochs is 1000
model.fit(training_input, training_target, validation_split=0.2,
          callbacks=[EarlyStopping(patience=10),
                     ModelCheckpoint('model.h5', verbose=1, save_best_only=True),
                     tensorboard],
          batch_size=1024, nb_epoch=1000)

model = load_model('model.h5')

still_mask = training_target[:, 0].astype(bool)
print('STILL accuracy:', model.evaluate(training_input[still_mask], training_target[still_mask], verbose=0)[1])
print('MOVE accuracy:', model.evaluate(training_input[~still_mask], training_target[~still_mask], verbose=0)[1])
