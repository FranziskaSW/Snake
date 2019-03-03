from policies import base_policy as bp
# from policies import DQNetwork as DQN
import numpy as np
from keras.models import Sequential
from keras.layers import *
import keras
import os
import pickle # TODO REMOVE
import math

from keras.models import Sequential
from keras.layers import *
import keras

global cwd # TODO remove
cwd = os.getcwd()


EPSILON = 0.2
EPSILON_RATE = 0.9999
LEARNING_RATE = 0.01

NUM_ACTIONS = 3  # (L, R, F)
VICINITY = 3
MAX_DISTANCE = 2
BATCH_SIZE = 64
GAMMA = 0.5
DROPOUT_RATE = 0.2


class DQNetwork():
    def __init__(self, input_shape, alpha, gamma,
                 dropout_rate, num_actions, batch_size, learning_rate, feature_num):
        self.alpha = alpha
        self.gamma = gamma
        self.dropout_rate = dropout_rate
        self.input_shape = input_shape
        self.feature_num = feature_num
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.act2idx = {'L': 0, 'R': 1, 'F': 2}

        self.model = Sequential()
        self.model.add(Dense(64, activation='relu', input_shape=self.input_shape))
        self.model.add(Dropout(dropout_rate))
        self.model.add(Dense(1))
        adam = keras.optimizers.Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        self.model.compile(loss='mean_squared_error', optimizer=adam)

    def learn(self, batches):
        bs_int = int(self.batch_size)
        x = np.zeros([bs_int, self.feature_num])
        y = np.zeros(bs_int)
        for idx, batch in enumerate(batches):
            x[idx] = batch['s_t']
            q_hat = (batch['r_t'] + self.gamma * np.max(self.model.predict(batch['s_tp1']))) # here need to do loop as well, which one of the 3 actions gives best q-value.
            y[idx] = q_hat

        h = self.model.fit(x, y, batch_size=bs_int, epochs=1, verbose=0)
        return h.history['loss'][0]

class MyPolicy(bp.Policy):
    """
    """

    def cast_string_args(self, policy_args):
        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
        policy_args['epsilon_rate'] = float(policy_args['epsilon_rate']) if 'epsilon_rate' in policy_args else EPSILON_RATE
        policy_args['batch_size'] = float(policy_args['batch_size']) if 'batch_size' in policy_args else BATCH_SIZE
        policy_args['vicinity'] = float(policy_args['vicinity']) if 'vicinity' in policy_args else VICINITY
        policy_args['learning_rate'] = float(policy_args['learning_rate']) if 'learning_rate' in policy_args else LEARNING_RATE
        policy_args['gamma'] = float(policy_args['gamma']) if 'gamma' in policy_args else GAMMA
        return policy_args

    def init_run(self):
        self.r_sum = 0
        self.vicinity = VICINITY
        self.max_distance = MAX_DISTANCE
        self.feature_num = (self.vicinity*2+1)**2*11 +1  # (self.max_distance+1+1)*11 + 1 + ((self.vicinity*2+1)**2*11)
        self.section_indices = np.array(range((self.vicinity*2+1)**2)) * 11
        self.input_shape = (self.feature_num, )
        self.Q = DQNetwork(input_shape=self.input_shape, alpha=0.5, gamma=0.8,
                           dropout_rate=DROPOUT_RATE, num_actions=NUM_ACTIONS, batch_size=self.batch_size,
                           learning_rate=self.learning_rate, feature_num=self.feature_num)
        self.memory = []
        self.loss = []
        self.act2idx = {'L': 0, 'R': 1, 'F': 2}
        self.idx2act = {0: 'L', 1: 'R', 2: 'F'}
        self.memory_length = int(self.batch_size*20)
        self.epsilon_rate = EPSILON_RATE
        self.not_too_slow_count = 0


    def put_stats(self):  # TODO remove after testing
        pickle.dump(self.loss, open(self.dir_name + '/last_game_loss.pkl', 'wb'))
        pickle.dump(self.test(), open(self.dir_name + '/last_test_loss.pkl', 'wb'))

    def test(self):  # TODO REMOVE AFTER TESTING
        loss = self.loss
        return loss


    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):

        if self.epsilon >= 0.1:
            self.epsilon = self.epsilon * self.epsilon_rate

        if round >= self.batch_size:
            bs_int = int(self.batch_size)
            random_batches = np.random.choice(self.memory, bs_int)
            loss = self.Q.learn(random_batches)
            self.loss.append(loss)

        try:
            if round % 100 == 0:
                if round > self.game_duration - self.score_scope:
                    self.log("Rewards in last 100 rounds which counts towards the score: " + str(self.r_sum), 'VALUE')
                else:
                    self.log("Rewards in last 100 rounds: " + str(self.r_sum), 'VALUE')
                self.r_sum = 0
            else:
                self.r_sum += reward

        except Exception as e:
            self.log("Something Went Wrong...", 'EXCEPTION')
            self.log(e, 'EXCEPTION')

    def getVicinityMap(self, board, center, direction):
        vicinity = self.vicinity

        r, c = center
        left = c - vicinity
        right = c + vicinity + 1
        top = r - vicinity
        bottom = r + vicinity + 1

        big_board = board

        if left < 0:
            left_patch = np.matrix(big_board[:, left])
            left = 0
            right = 2 * vicinity + 1
            big_board = np.hstack([left_patch.T, big_board])

        if right >= board.shape[1]:
            right_patch = np.matrix(big_board[:, :(right % board.shape[1] + 1)])
            big_board = np.hstack([big_board, right_patch])

        if top < 0:
            top_patch = np.matrix(big_board[top, :])
            top = 0
            bottom = 2 * vicinity + 1
            big_board = np.vstack([top_patch, big_board])

        if bottom >= board.shape[0]:
            bottom_patch = np.matrix(big_board[:(bottom % board.shape[0])])
            big_board = np.vstack([big_board, bottom_patch])

        map = big_board[top:bottom, left:right]

        if direction == 'N': return map
        if direction == 'E': return np.rot90(map, k=1)
        if direction == 'S': return np.rot90(map, k=2)
        if direction == 'W': return np.rot90(map, k=-1)

    def getVicinityRepresentation(self, VicinityMap):
        symbols = np.matrix(VicinityMap.flatten())
        symbols = symbols[0]
        features = np.zeros(symbols.shape[1]*11)

        section_indices = self.section_indices
        symbol_idx = symbols + section_indices + 1
        symbol_idx_list = symbol_idx.tolist()[0]
        idx_int = [int(x) for x in symbol_idx_list]
        features[idx_int] = 1
        return features

    def getFeature_2(self, board, head, action):
        head_pos, direction = head
        moving_dir = bp.Policy.TURNS[direction][action]
        next_position = head_pos.move(moving_dir)
        map_v = self.getVicinityMap(board, next_position, moving_dir)
        features_1 = np.array((board==self.id).sum())
        features_2 = self.getVicinityRepresentation(map_v)
        features = np.append(features_1, features_2)
        features = features/features.sum()
        return features

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):

        if too_slow & (round >= 100):
            self.batch_size = math.ceil(self.batch_size/2)

            self.log(str(self.id) + "lower batch size to " + str(self.batch_size), 'action')  # TODO: remove
            self.not_too_slow_count = 0
        else:
            self.not_too_slow_count += 1

        if (self.not_too_slow_count == 200) & (self.batch_size < BATCH_SIZE):
            self.log("reset batch size to " + str(self.batch_size*2), 'action')  # TODO: remove
            self.batch_size = int(self.batch_size*2)


        board, head = new_state
        new_features = np.zeros([len(self.act2idx), self.feature_num])

        random_actions = np.random.permutation(bp.Policy.ACTIONS)
        for i, a in enumerate(random_actions):
            new_features[i] = self.getFeature_2(board, head, a)

        if round >=2:  # update to memory from previous round (prev_state)
            prev_board, prev_head = prev_state
            prev_feature = self.getFeature_2(prev_board, prev_head, prev_action)
            memory_update = {'s_t': prev_feature, 'a_t': self.act2idx[prev_action], 'r_t': reward, 's_tp1': new_features}

            if len(self.memory) < self.memory_length:
                self.memory.append(memory_update)
            else:
                idx = np.random.choice(range(0, self.memory_length))
                self.memory[idx] = memory_update

        if round == (self.game_duration-1):  # TODO: remove before assignment
            losses = self.loss
            with open(cwd + "/losses.pickle", "wb") as f:
                pickle.dump(losses, f)

        # act in new round, decide for new_state
        if (np.random.rand() < self.epsilon) & (round < self.game_duration - self.score_scope):
            action = np.random.choice(bp.Policy.ACTIONS)

        else:
            q_values = self.Q.model.predict(new_features, batch_size=3)
            a_idx = np.argmax(q_values)
            action = random_actions[a_idx]

        return action
