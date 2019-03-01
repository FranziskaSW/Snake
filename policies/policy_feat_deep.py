from policies import base_policy as bp
import numpy as np
import random
from copy import deepcopy  #
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Flatten, Dropout
from keras.callbacks import TensorBoard
from time import time
# from Snake import Position
from keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt
import sys
from time import time

import pickle  # TODO remove after testing

LEARNING_TIME = bp.LEARNING_TIME

NUM_SYMBOLS = 11
DIRECTIONS = ('N', 'W', 'E', 'S')

# Default parameters:
BATCH_SIZE = 16
EPSILON = 0.3
GAMMA = 0.5
LEARNING_RATE = 0.0001
SPATIAL_BOX_RADIUS = 2
DISTANCE_SCOPE_RADIUS = 5
INITIAL_EXPLORATION_TIME = 100


class FeatDeep(bp.Policy):
    """
    A deep Q-Learning approximation which avoids collisions with obstacles and other snakes ad 
    wish to eat good yummy fruits. It has an epsilon parameter which controls the
    percentag of actions which are randomly chosen.
    """
    def put_stats(self):  # TODO remove after testing
        pickle.dump(self.loss, open(self.dir_name + '/last_game_loss.pkl', 'wb'))
        pickle.dump(self.test(), open(self.dir_name + '/last_test_loss.pkl', 'wb'))

    def cast_string_args(self, policy_args):
        policy_args['epsilon'] = float(
            policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
        policy_args['gamma'] = float(
            policy_args['gamma']) if 'gamma' in policy_args else GAMMA
        policy_args['lr'] = float(
            policy_args['lr']) if 'lr' in policy_args else LEARNING_RATE
        policy_args['bs'] = int(
            policy_args['bs']) if 'bs' in policy_args else BATCH_SIZE
        policy_args['spatial_r'] = int(
            policy_args['spatial_r']) if 'spatial_r' in policy_args else SPATIAL_BOX_RADIUS
        policy_args['dist_r'] = int(
            policy_args['dist_r']) if 'dist_r' in policy_args else DISTANCE_SCOPE_RADIUS
        policy_args['exploration_t'] = int(
            policy_args['exploration_t']) if 'exploration_t' in policy_args else INITIAL_EXPLORATION_TIME

        return policy_args

    def init_run(self):
        self.loss = []
        self.reward_avg = []
        self.r_sum = 0
        self.act2idx = {'L': 0, 'R': 1, 'F': 2}
        self.idx2act = {0: 'L', 1: 'R', 2: 'F'}
        self.max_replay = 2000  # todo calculate according to net size and board size
        self.distance_scope = min(self.dist_r, min(self.board_size[0], self.board_size[1]) // 2)
        self.spatial_box = min(self.spatial_r, (min(self.board_size[0], self.board_size[1]) - 1) // 2)
        box_feature_num = (((self.spatial_box * 2) + 1) ** 2) * NUM_SYMBOLS
        self.vicinity = 8
        self.n_features = 10 * (self.vicinity*2 + 2)  # ((self.distance_scope + 1) * NUM_SYMBOLS) + 1 + box_feature_num

        # todo calc
        self.replay_prev = np.zeros(shape=(self.max_replay, self.n_features))
        self.replay_next = np.zeros(shape=(self.max_replay, 3, self.n_features))
        self.replay_reward = np.zeros(shape=(self.max_replay))
        self.replay_action = np.zeros(shape=(self.max_replay), dtype=np.int8)
        self.replay_idx = 0
        self.first_act = True
        self.rotate_num = {'N': 0, 'E': 1, 'S': 2, 'W': 3}
        self.init_roll_params()
        self.reward_stats = 0
        self.reward_count = 0

        print("FEATURES NUM:", self.n_features)
        print('GAME DUR:', self.game_duration)

        # old features
        '''
        # distance features - codes which (and how many) symbols will be in d-steps distance after an action

        # self.distance_scope_mask = self.init_distance_masks()

        # distant features - codes a general description of what happens far away
        # self.distant_scope = max(self.board_size // 4, 30) # todo board size
        # self.is_distant_min = 5
        # +1 for also considering the board after any action from current state
        # self.distant_scope_mask = self.create_dist_mask(dist=self.distant_scope+1) > self.is_distant_min

        # # spatial features - codes exactly is going to be in a d-ball around the head of the snake after an action
        # self.spatial_scope = min(self.board_size // 2, 3)  # todo flatten by dir
        # self.spatial_features_size = (2 * self.spatial_scope + 1) ** 2
        # self.spatial_features_range = np.arange(self.spatial_features_size)
        # self.spatial_scope += 1  # +1 for also considering the board after any action from current state

        # placholder for additional features: length of snake, etc.
        '''

        self.init_net()

    def init_roll_params(self):
        self.roll_params = dict()
        h, w = self.board_size
        hight_even = 1 - int(h % 2)
        width_even = 1 - int(w % 2)

        # if we have north
        r = h // 2  # center row index
        c = w // 2  # center column index
        above = r
        below = r - hight_even
        left = c
        right = c - width_even
        north = {'r': r, 'c': c, 'above': above, 'below': below, 'left': left, 'right': right,
                 'box': (r - 2, r + 2, c)}
        self.roll_params['N'] = north

        # if we have south
        # if h is odd so same as north, else every row should roll up by 1
        south = deepcopy(north)
        south['r'] -= hight_even
        south['above'] -= hight_even
        south['below'] += hight_even
        self.roll_params['S'] = south

        # west
        west = deepcopy(north)
        self.roll_params['W'] = west

        # if we have east
        east = deepcopy(west)
        east['c'] -= width_even
        east['left'] -= width_even
        east['right'] += width_even
        self.roll_params['E'] = east

        for dir in self.roll_params:
            r, c = self.roll_params[dir]['r'], self.roll_params[dir]['c']
            distance = self.get_distance_from_location((r, c), dir)
            masks = dict()
            for d in range(self.distance_scope):
                masks[d] = np.where(distance == d)
            masks[d + 1] = np.where(distance >= self.distance_scope)
            self.roll_params[dir]['distance_masks'] = masks

    def init_net(self):
        # input size
        self.model = Sequential([
            Dense(256, activation='relu', input_shape=(self.n_features,)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            # Dropout(0.3),
            # Dense(self.n_features, activation='relu'),
            Dropout(0.3),
            Dense(1),
        ])
        adam = Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        # sgd = SGD(lr=self.lr)
        self.model.compile(loss='mean_squared_error', optimizer=adam)

        # self.tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

        self.warmup()

    def test(self):  # TODO REMOVE AFTER TESTING
        gt = self.prep_ground_truth(self.replay_next, self.replay_reward, self.replay_idx)
        loss = self.model.train_on_batch(self.replay_prev[:self.replay_idx], gt)
        return loss

    def warmup(self):  # TODO try to make warmup effective for the first 100 rounds problem, otherwise delete
        for i in range(2000):
            # save into replay buffer:
            #  board = np.random.randint(low=-1, high=11, size=(self.board_size[0], self.board_size[1]))
            # # head = Position((np.random.randint(self.board_size[0], self.board_size[1])),
            # #                 self.board_size)

            self.replay_prev[i] = np.random.randint(low=-1, high=11, size=(self.n_features))
            self.replay_next[i] = np.random.randint(low=-1, high=11, size=(3, self.n_features))
            act = self.idx2act[np.random.randint(0, 3)]
            self.replay_action[i] = self.act2idx[act]
            self.replay_reward[i] = np.random.randint(-5, 6)
            net_input = np.random.randint(-1, 11,
                                          size=(self.bs, self.n_features))
            net_output = self.model.predict(net_input, batch_size=self.bs, verbose=0)
            # print('warm up', i)

    def prep_ground_truth(self, next_features, rewards, batch_size=None):  # TODO vectorize
        if batch_size is None:
            batch_size = self.bs
        gt = np.zeros(batch_size)
        for i in range(batch_size):
            net_input = next_features[i]
            net_output = self.model.predict(net_input, batch_size=3)
            gt[i] = self.gamma*np.max(net_output) + rewards[i]
        return gt

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        replace = self.replay_idx < self.bs
        # print('replay_idx', self.replay_idx)
        choice = np.random.choice(range(self.replay_idx), size=self.bs, replace=replace)
        # print('choice', choice)
        prev_batch = self.replay_prev[choice]
        # print('prev batch', prev_batch.shape)
        next_batch = self.replay_next[choice]
        reward_batch = self.replay_reward[choice]
        action_batch = self.replay_action[choice]

        gt = self.prep_ground_truth(next_batch, reward_batch)

        self.loss.append(self.model.train_on_batch(prev_batch, gt))
        self.reward_avg.append(self.reward_stats / self.reward_count)

        self.reward_stats = self.reward_count = 0.

        # self.model.fit(prev_batch, target, self.bs, epochs=1, verbose=0, callbacks=[self.tensorboard])

        # if round > self.game_duration - 6:
        #     plt.plot(range(len(self.loss)), self.loss)
        #     smooth = list(range(0, len(self.loss), 15))
        #     plt.plot(smooth, np.array(self.loss)[smooth])
        #     plt.savefig('loss_s.png')
        #     plt.clf()
        #     plt.plot(range(len(self.reward_avg)), self.reward_avg)
        #     plt.plot(range(len(self.reward_avg)), [0] * len(self.reward_avg))
        #     plt.savefig('reward_s.png')
        #     plt.clf()

        try:
            if round % 100 == 0:
                if round > self.game_duration - self.score_scope:
                    self.log("Rewards in last 100 rounds which counts towards the score: " + str(
                        self.r_sum), 'VALUE')
                else:
                    self.log("Rewards in last 100 rounds: " + str(self.r_sum), 'VALUE')

                self.r_sum = 0
            else:
                self.r_sum += reward

        except Exception as e:
            self.log("Something Went Wrong...", 'EXCEPTION')
            self.log(e, 'EXCEPTION')

    def get_distance_from_location(self, location, direction):
        """
        Calculates the city block distance from a certain location, taking into account that the 
        board is cyclic and only L,R,F actions are allowed
        :param location: tuple of rows,cols - location on the board
        :param direction: one of 'N', 'S', 'E', 'W'
        :return: the distances
        """
        h, w = self.board_size
        cols_idx, rows_idx = np.meshgrid(range(w), range(h))
        down_idx, up_idx = rows_idx, np.flip(rows_idx, axis=0)
        right_idx, left_idx = cols_idx, np.flip(cols_idx, axis=1)

        loc_r, loc_c = location

        if direction == 'N':
            down_idx[1:, loc_c] += 2
        elif direction == 'S':
            up_idx[:-1, loc_c] += 2
        elif direction == 'E':
            left_idx[loc_r, :-1] += 2
        elif direction == 'W':
            right_idx[loc_r, 1:] += 2

        roll_down = np.roll(down_idx, loc_r, axis=0)
        roll_up = np.roll(up_idx, -(h - loc_r - 1), axis=0)
        vertical_distance = np.minimum(roll_down, roll_up)

        roll_right = np.roll(right_idx, loc_c, axis=1)
        roll_left = np.roll(left_idx, -(w - loc_c - 1), axis=1)
        horizontal_distance = np.minimum(roll_left, roll_right)

        distance = vertical_distance + horizontal_distance
        return distance

    def snake_length(self, state):
        """
        returns the normalized length of this snake in the current state.
        """
        board, head = state
        head_pos, direction = head
        self_symbol = board[head_pos[0], head_pos[1]]
        length = np.count_nonzero(board == self_symbol)
        # normalize length
        return np.array([length / float(board.size)])

    def spatial_features(self, rolled_board, dir):
        r, c = self.roll_params[dir]['r'], self.roll_params[dir]['c']
        b = self.spatial_box
        box = rolled_board[r - b:r + b + 1, c - b:c + b + 1]
        box = np.rot90(box, k=self.rotate_num[dir]).flatten()
        assert box.size == ((b * 2) + 1) ** 2
        spatial_features = np.zeros(NUM_SYMBOLS * box.size)
        box_indices = np.array(range(box.size))*NUM_SYMBOLS
        spatial_features[box_indices+box+1] = 1
        return spatial_features

    def distance_features(self, state, action):
        """
        Calculate the distance features- for each distance d in self.distance_scope, we count how
        many symbols of each type appear on the board with city block distance d from the 
        location after taking the specified action.
        """
        board, head = state
        head_pos, direction = head
        next_direction = bp.Policy.TURNS[direction][action]
        next_location = head_pos.move(next_direction)
        r = next_location[0]
        c = next_location[1]

        # roll board
        rolled_board = np.roll(board, self.roll_params[next_direction]['c'] - c, axis=1)
        rolled_board = np.roll(rolled_board, self.roll_params[next_direction]['r'] - r, axis=0)

        # distance = self.get_distance_from_location((r, c), next_direction)
        masks = self.roll_params[next_direction]['distance_masks']
        features = np.zeros((self.distance_scope + 1) * NUM_SYMBOLS)

        for d in range(self.distance_scope + 1):
            f = np.bincount(rolled_board[masks[d]] + 1, minlength=NUM_SYMBOLS)
            # assert np.sum(f) == len(rolled_board[np.where(distance == d)])
            # print(f.shape, np.sum(f))
            # sys.exit()
            f = f / np.sum(f)  # TODO no normalization?
            features[d*NUM_SYMBOLS:(d+1)*NUM_SYMBOLS] = f
            # features[d:d + NUM_SYMBOLS] = f  # == previous code
            # print('d', d, time()-s)

        # return features
        return np.concatenate((features, self.spatial_features(rolled_board, next_direction)))

    def get_features(self, state, action):
        return np.concatenate((self.distance_features(state, action), self.snake_length(state)))

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

    def getFeature_2(self, board, head, action):
        head_pos, direction = head
        moving_dir = bp.Policy.TURNS[direction][action]
        next_position = head_pos.move(moving_dir)
        map_v = self.getVicinityMap(board, next_position, moving_dir)
        center = (self.vicinity, self.vicinity)
        max_distance = self.vicinity * 2
        features = np.zeros(self.n_features)

        for field_value in range(-1, 10):
            feature_idx = int(field_value) + 1
            # how many elements do we have
            features[feature_idx] = (board == field_value).sum()

            m = (map_v == field_value)
            field_positions = np.matrix(np.where(m)).T

            distances = []
            for field_pos in field_positions:
                x, y = field_pos.tolist()[0][0], field_pos.tolist()[0][1]
                dist = abs(center[0] - x) + abs(center[1] - y)
                distances.append(dist)
            # fill feature vector
            for val in range(0, max_distance + 1):
                if val in distances:
                    idx_area = val + 1
                    idx = feature_idx + idx_area * 10
                    features[idx] = 1

        return features

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):

        if round > self.game_duration - self.score_scope:  # TODO change with decaying epsilon
            eps = 0.0  # todo handle decreasing lr
        elif round <= self.exploration_t:
            eps = 1.
        else:
            eps = self.epsilon

        if not self.first_act:
            self.reward_stats += reward
            self.reward_count += 1
            idx = self.replay_idx
            if idx >= self.max_replay:
                idx = random.randint(0, self.max_replay - 1)
            else:
                # print('increase replay idx')
                self.replay_idx += 1

            # save into replay buffer:
            prev_board, prev_head = prev_state
            self.replay_prev[idx] = self.getFeature_2(prev_board, prev_head, prev_action)# self.get_features(prev_state, prev_action)
            # self.replay_prev[idx] = np.random.randint(low=0, high=11, size=(self.n_features))
            board, head = new_state
            next_features = np.zeros((3, self.n_features))
            for a in self.act2idx:
                next_features[self.act2idx[a]] = self.getFeature_2(board, head, a) # self.get_features(new_state, a)
            # next_features = np.random.randint(low=0, high=11, size=(3, self.n_features))
            self.replay_next[idx] = next_features
            self.replay_action[idx] = self.act2idx[prev_action]
            self.replay_reward[idx] = reward

        if np.random.rand() <= eps or self.first_act:
            self.first_act = False
            return np.random.choice(bp.Policy.ACTIONS)

        # print('features: ', next_features.shape)
        # net_input = self.replay_next[idx][np.newaxis, ...]
        net_input = self.replay_next[idx]
        net_output = self.model.predict(net_input, batch_size=3)
        action_idx = np.argmax(net_output)
        action = self.idx2act[action_idx]
        # print('CHOSE ACT:', action)
        return action

    @staticmethod
    def create_dist_mask(dist):
        """
        Builds a (2*dist + 1, 2*dist + 1) distance transform matrix with d1 distance (city block distance)
        from the center point (dist, dist).
        """
        x_arr, y_arr = np.mgrid[0:2 * dist + 1, 0:2 * dist + 1]
        cell = (dist, dist)
        dist_mask = abs(x_arr - cell[0]) + abs(y_arr - cell[1])
        return dist_mask

    @staticmethod
    def pad_board(board, distance, r, c):
        """
        Pads the board cyclically if needed, according to required distance from the point (r, c),
        so that the square [r-distance, r+distance] X [c-distance, c+distance] would be completely
        contained in the padded board.
        :param board: board (2D numpy array) to pad
        :param distance: distance from (r, c)
        :param r: rows coordinate
        :param c: columns coordinate
        :return: padded board
        """
        w, h = board.shape
        up_pad = max(0, distance - r)
        down_pad = max(0, distance - (h - r))
        right_pad = max(0, distance - (w - c))
        left_pad = max(0, distance - c)
        return np.pad(board, ((up_pad, down_pad), (left_pad, right_pad)), mode='wrap')

    # old methods:
    '''
    def init_distance_masks(self):
        """
        returns a dictionary of boolean masks, indicating which cells in the scope are in city-block distance
        of exactly d from the center point of a (2*self.distance_scope + 1, 2*self.distance_scope + 1) matrix.
        """
        basic_masks = dict()
        basic_masks['N'] = self.create_dist_mask(self.distance_scope)
        basic_masks['S'] = basic_masks['N'].copy()
        basic_masks['W'] = basic_masks['N'].copy()
        basic_masks['E'] = basic_masks['N'].copy()

        basic_masks['N'][self.distance_scope + 1:, self.distance_scope] += 2
        basic_masks['S'][:self.distance_scope, self.distance_scope] += 2
        basic_masks['W'][self.distance_scope, self.distance_scope + 1:] += 2
        basic_masks['E'][self.distance_scope, :self.distance_scope] += 2

        result = dict()
        for direction in DIRECTIONS:
            result[direction] = dict()
            for d in range(1, self.distance_scope + 1):
                result[direction][d] = basic_masks[direction] == d

        return result
    
    def features(self, state, action=None):
        """
        returns the feature vector for the given state and action.
        if action is not supplied, returns an array of feature vectors with one for each possible action.
        :param state: current state
        :param action: chosen action
        :return: feature vector(s)
        """
        distance_features = self.distance(state, action)
        spatial_features = self.spatial(state, action)
        distant_features = self.distant(state, action)

        if action is None:  # when features are used for predict
            features = np.zeros((len(self.ACTIONS), self.n_features))
            for action in self.ACTIONS:
                action_i = self.act2idx[action]
                feature_i = 0
                for partial_features in [distance_features[action_i],
                                         spatial_features[action_i],
                                         distant_features[action_i]]:
                    end = len(partial_features)
                    features[action_i][feature_i:feature_i + end] = partial_features
                    feature_i += end
            return features

        else:
            features = np.zeros(self.n_features)
            feature_i = 0
            # todo concat
            for partial_features in [distance_features,
                                     spatial_features,
                                     distant_features]:
                end = len(partial_features)
                features[feature_i:feature_i + end] = partial_features
                feature_i += end
            return features


    def distance(self, state,
                 action=None):  # TODO - Hi Gal! this is how I continued to write the distance features extraction
        """
        Calculates distance features.
        :param state: state for which to calculate features.
        :param action: action for which to calculate features. if None, calculates features for all actions.
        :return: a (NUM_SYMBOLS,) feature vector if action is specified,
                  a (len(self.ACTIONS), NUM_SYMBOLS) array of feature vectors otherwise.
        """
        if action is not None:
            total_scope = 2 * self.distance_scope + 1
            board, head = state
            head_pos, direction = head
            next_position = head_pos.move(bp.Policy.TURNS[direction][action])
            r = next_position[0]
            c = next_position[1]

            wrapped_board = self.pad_board(board, self.distance_scope, r, c)
            scoped_board = wrapped_board[r:(r + total_scope), c:(c + total_scope)]

            return self.get_distance_features(scoped_board, direction)

        else:
            # +1 for also considering the board after any action from current state:
            total_scope = 2 * (self.distance_scope + 1) + 1
            board, head = state
            head_pos, direction = head
            r = head_pos[0]
            c = head_pos[1]

            wrapped_board = self.pad_board(board, self.distance_scope + 1, r, c)
            scoped_board = wrapped_board[r:(r + total_scope), c:(c + total_scope)]

            features = np.zeros((len(self.ACTIONS), (self.distance_scope + 1) * NUM_SYMBOLS))
            for action in self.ACTIONS:
                next_direction = self.TURNS[direction][action]
                if next_direction == 'S':
                    features[self.act2idx[action]] = self.get_distance_features(
                        scoped_board[2:, 1:-1], direction)
                elif next_direction == 'N':
                    features[self.act2idx[action]] = self.get_distance_features(
                        scoped_board[:-2, 1:-1], direction)
                elif next_direction == 'E':
                    features[self.act2idx[action]] = self.get_distance_features(
                        scoped_board[1:-1, 2:], direction)
                else:  # next_direction == 'W'
                    features[self.act2idx[action]] = self.get_distance_features(
                        scoped_board[1:-1, :-2], direction)
            return features

    def get_distant_features(self, scoped_board, direction, action):
        """
        Calculates distant features for a certain direction.
        :param scoped_board: scope of board for which to calculate features.
        :param direction: direction of head in current state.
        :param action: action for which to calculate features.
        :return: a (NUM_SYMBOLS,) feature vector.
        """
        r = c = self.distant_scope  # coordinates of head in current state, in scoped board
        next_direction = self.TURNS[direction][action]
        # np.bincount is only for non-negative numbers, so we use +1 (one-to-one and retains order)
        if next_direction == 'S':
            return np.bincount(scoped_board[r + 2:, :][self.distant_scope_mask[r + 2:, :]] + 1)
        elif next_direction == 'N':
            return np.bincount(scoped_board[:r - 1, :][self.distant_scope_mask[:r - 1, :]] + 1)
        elif next_direction == 'E':
            return np.bincount(scoped_board[:, c - 2:][self.distant_scope_mask[:, c - 2:]] + 1)
        else:  # next_direction == 'W'
            return np.bincount(scoped_board[:, :c - 1][self.distant_scope_mask[:, :c - 1]] + 1)

    def distant(self, state, action=None):
        """
        Calculates distant features.
        :param state: state for which to calculate features.
        :param action: action for which to calculate features. if None, calculates features for all actions.
        :return: a (NUM_SYMBOLS,) feature vector if action is specified,
                  a (len(self.ACTIONS), NUM_SYMBOLS) array of feature vectors otherwise.
        """
        total_scope = 2 * self.distant_scope + 1
        board, head = state
        head_pos, direction = head
        r = head_pos[0]
        c = head_pos[1]
        wrapped_board = self.pad_board(board, self.distant_scope, r, c)
        scoped_board = wrapped_board[r:(r + total_scope), :(c + total_scope)]

        if action is None:
            features = np.zeros((len(self.ACTIONS), NUM_SYMBOLS))
            for action in self.ACTIONS:
                features[self.act2idx[action]] = self.get_distant_features(scoped_board, direction,
                                                                           action)
            return features
        else:
            return self.get_distant_features(scoped_board, direction, action)

    def get_spatial_features(self, scoped_board, direction, action):
        """
        Calculates spatial features for a certain direction.
        :param scoped_board: scope of board for which to calculate features.
        :param direction: direction of head in current state.
        :param action: action for which to calculate features.
        :return: a (NUM_SYMBOLS,) feature vector.
        """
        r = c = self.spatial_scope  # coordinates of head in current state, in scoped board
        next_direction = self.TURNS[direction][action]
        if next_direction == 'S':
            scoped_board = scoped_board[2:, 1:-1].flatten() + 1
        elif next_direction == 'N':
            scoped_board = scoped_board[:-2, 1:-1].flatten() + 1
        elif next_direction == 'E':
            scoped_board = scoped_board[1:-1, 2:].flatten() + 1
        else:  # next_direction == 'W'
            scoped_board = scoped_board[1:-1, :-2].flatten() + 1
        features = np.zeros((self.spatial_features_size, NUM_SYMBOLS))

        features[
            self.spatial_features_range, scoped_board] = 1  # TODO: this is the fancy indexing indexing I was talking about (delete this)
        return features.flatten()
    
    def get_distance_features(self, scoped_board, next_direction):
        """
        Calculates distance features for a certain direction.
        :param scoped_board: scope of board for which to calculate features.
        :param next_direction: direction of head after action.
        :return: a ((self.distance_scope+1) * NUM_SYMBOLS,) feature vector.
        """
        r = c = self.distance_scope  # coordinates of head in current state, in scoped board
        features = np.zeros((self.distance_scope + 1, NUM_SYMBOLS))
        # np.bincount is only for non-negative numbers, so we use +1 (one-to-one and retains order)
        for d in range(0, self.distance_scope + 1):
            if next_direction == 'S':
                features[d] = np.bincount(scoped_board[self.distant_scope_mask['S'][d]] + 1)
            elif next_direction == 'N':
                features[d] = np.bincount(scoped_board[self.distant_scope_mask['N'][d]] + 1)
            elif next_direction == 'E':
                features[d] = np.bincount(scoped_board[self.distant_scope_mask['E'][d]] + 1)
            else:  # next_direction == 'W'
                features[d] = np.bincount(scoped_board[self.distant_scope_mask['W'][d]] + 1)
        return features.flatten()    

    def spatial(self, state, action=None):
        """
        Calculates spatial features.
        :param state: state for which to calculate features.
        :param action: action for which to calculate features. if None, calculates features for all actions.
        :return: a (NUM_SYMBOLS,) feature vector if action is specified,
                  a (len(self.ACTIONS), NUM_SYMBOLS) array of feature vectors otherwise.
        """
        total_scope = 2 * self.spatial_scope + 1

        board, head = state
        head_pos, direction = head
        r = head_pos[0]
        c = head_pos[1]
        wrapped_board = self.pad_board(board, self.spatial_scope, r, c)
        scoped_board = wrapped_board[r:(r + total_scope), c:(c + total_scope)]

        if action is None:
            features = np.zeros((len(self.ACTIONS), self.spatial_features_size * NUM_SYMBOLS))
            for action in self.ACTIONS:
                features[self.act2idx[action]] = self.get_spatial_features(scoped_board, direction,
                                                                           action)
            return features
        else:
            return self.get_spatial_features(scoped_board, direction, action)
    '''
