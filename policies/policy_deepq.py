from policies import base_policy as bp
from policies import DQNetwork as DQN
import numpy as np
from keras.models import Sequential
from keras.layers import *
import keras
import os
import pickle # TODO REMOVE

global cwd
cwd = os.getcwd()


EPSILON = 0.2
EPSILON_RATE = 0.9999
LEARNING_RATE = 0.0001

NUM_ACTIONS = 3  # (L, R, F)
VICINITY = 3
MAX_DISTANCE = 2
BATCH_SIZE = 32
GAMMA = 0.7
DROPOUT_RATE = 0.2

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
        self.Q = DQN.DQNetwork(input_shape=self.input_shape, alpha=0.5, gamma=0.8,
                               dropout_rate=DROPOUT_RATE, num_actions=NUM_ACTIONS, batch_size=self.batch_size,
                               learning_rate=self.learning_rate, feature_num=self.feature_num)
        self.memory = []
        self.loss = []
        self.act2idx = {'L': 0, 'R': 1, 'F': 2}
        self.idx2act = {0: 'L', 1: 'R', 2: 'F'}
        self.memory_length = int(self.batch_size*20)


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
        self.epsilon = self.epsilon * EPSILON_RATE

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

    def getFeature(self, board, head, action):
        head_pos, direction = head
        moving_dir = bp.Policy.TURNS[direction][action]
        next_position = head_pos.move(moving_dir)
        map_after = self.getVicinityMap(board, next_position, moving_dir)
        features = map_after.flatten()
        return features
    #
    # def getFeature_2(self, board, head, action):
    #     head_pos, direction = head
    #     moving_dir = bp.Policy.TURNS[direction][action]
    #     # next_position = head_pos.move(moving_dir)
    #     next_position = (5, 5)
    #     map_v = self.getVicinityMap(board, next_position, moving_dir)
    #     center = (self.vicinity, self.vicinity)
    #     features = np.zeros(self.feature_num)
    #
    #     for field_value in range(-1, 10):
    #         # # how many elements do we have in vicinity
    #         # offset = 1
    #         # feature_idx = int(field_value) + offset
    #         # features[feature_idx] = (map_v == field_value).sum()
    #
    #         # how long are we
    #         features[-1] = (board == self.id).sum()
    #
    #         # what is in next and second next field?
    #         m = (map_v == field_value)
    #         field_positions = np.matrix(np.where(m)).T
    #
    #         distances = []
    #         for field_pos in field_positions:
    #             x, y = field_pos.tolist()[0][0], field_pos.tolist()[0][1]
    #             dist = abs(center[0] - x) + abs(center[1] - y)
    #             distances.append(dist)
    #
    #         # fill feature vector
    #         for dist in range(0, self.max_distance + 1):
    #             offset = 12
    #             if dist in distances:
    #                 idx = int(field_value) + (dist * 11) + offset
    #                 features[idx] = 1
    #
    #         # f = np.hstack([features, map_v.flatten()])
    #         print(features)
    #         # features = features/features.sum()
    #
    #     return features

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
        center = (self.vicinity, self.vicinity)

        # features_1 = np.zeros((self.max_distance+1+1)*11 + 1)
        #
        # # how long are we
        # features_1[0] = (board == self.id).sum()
        #
        # for field_value in range(-1, 10):
        #     # how many elements do we have in vicinity
        #     offset = 2
        #     feature_idx = int(field_value) + offset
        #     features_1[feature_idx] = (map_v == field_value).sum()
        #
        #     m = (map_v == field_value)
        #     field_positions = np.matrix(np.where(m)).T
        #
        #     distances = []
        #     for field_pos in field_positions:
        #         x, y = field_pos.tolist()[0][0], field_pos.tolist()[0][1]
        #         dist = abs(center[0] - x) + abs(center[1] - y)
        #         distances.append(dist)
        #
        #     # fill feature vector
        #     for dist in range(0, self.max_distance + 1):
        #         offset = 13
        #         if dist in distances:
        #             idx = int(field_value) + (dist * 11) + offset
        #             features_1[idx] = 1
        features_1 = np.array((board==self.id).sum())
        features_2 = self.getVicinityRepresentation(map_v)
        # print('features1 shape 45, features 2 shape 539 ', features_1.shape, features_2.shape)
        features = np.append(features_1, features_2)
        # print('features 539 + 45 ',features.shape)
        features = features/features.sum()
        return features

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        board, head = new_state
        new_features = np.zeros([len(self.act2idx), self.feature_num])

        for a in self.act2idx:
            new_features[self.act2idx[a]] = self.getFeature_2(board, head, a)

        if round >=2:  # update to memory from previous round (prev_state)
            prev_board, prev_head = prev_state
            prev_feature = self.getFeature_2(prev_board, prev_head, prev_action)
            memory_update = {'s_t': prev_feature, 'a_t': self.act2idx[prev_action], 'r_t': reward, 's_tp1': new_features}

            if len(self.memory) < self.memory_length:
                self.memory.append(memory_update)
            else:
                idx = np.random.choice(range(0, self.memory_length))
                self.memory[idx] = memory_update

        if round == (self.game_duration-1):
            losses = self.loss
            with open(cwd + "/losses.pickle", "wb") as f:
                pickle.dump(losses, f)

        # act in new round, decide for new_state
        if (np.random.rand() < self.epsilon) & (round < self.game_duration - self.score_scope):
            action = np.random.choice(bp.Policy.ACTIONS)

        else:
            q_values = self.Q.model.predict(new_features, batch_size=3)
            a_idx = np.argmax(q_values)
            action = self.idx2act[a_idx]

        return action
