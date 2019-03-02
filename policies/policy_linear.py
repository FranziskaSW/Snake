from policies import base_policy as bp
import numpy as np
import operator
import pickle  # TODO remove after testing

EPSILON = 0.3
EPSILON_RATE = 0.9999
GAMMA = 0.8
LEARNING_RATE = 0.01
LEARNING_TIME = bp.LEARNING_TIME

FEATURE_NUM = 11  # len(['Food1', 'Food2', 'Food3', 'FieldEmpty'])

from policies import base_policy as bp
import numpy as np

LEARNING_TIME = bp.LEARNING_TIME

EPSILON = 0.05
FEATURE_NUM = 11  # todo insert into dynamic func
GAMMA = 0.8
LEARNING_RATE = 0.01


class Linear_old(bp.Policy):
    """
    A policy which avoids collisions with obstacles and other snakes. It has an epsilon parameter which controls the
    percentag of actions which are randomly chosen.
    """

    def cast_string_args(self, policy_args):
        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
        policy_args['gamma'] = float(policy_args['gamma']) if 'gamma' in policy_args else GAMMA
        policy_args['learning_rate'] = float(policy_args['lr']) if 'lr' in policy_args else LEARNING_RATE

        return policy_args

    def init_run(self):
        # self.r_sum = 0
        # weights = np.matrix(np.random.uniform(0, 1, FEATURE_NUM))
        # self.weights = weights / weights.sum()
        # # self.last_actions = []  #
        # # self.last_qvalues = []
        # self.last_deltas = []
        # self.last_features = []
        # self.loss = [0]  # TODO  remove
        # self.epsilon_rate = EPSILON_RATE
        self.act2idx = {'L': 0, 'R': 1, 'F': 2}
        self.idx2act = {0: 'L', 1: 'R', 2: 'F'}
        self.feature_num = FEATURE_NUM
        self.r_sum = 0
        self.buffi = 0
        self.feature_buff = np.zeros(shape=(LEARNING_TIME * 2, FEATURE_NUM))
        self.deltas = np.zeros(shape=(LEARNING_TIME * 2, 1))
        self.weights = np.zeros(FEATURE_NUM)  # missin bias
        self.first_act = True
        self.epsilon_rate = EPSILON_RATE
        print('learning time: ', bp.LEARNING_TIME)

    def put_stats(self):  # TODO remove after testing
        pickle.dump(self.loss, open(self.dir_name + '/last_game_loss.pkl', 'wb'))
        pickle.dump(self.test(), open(self.dir_name + '/last_test_loss.pkl', 'wb'))

    def test(self):  # TODO REMOVE AFTER TESTING
        loss = self.loss
        return loss

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):

        if self.epsilon >= 0.1:
            self.epsilon = self.epsilon * self.epsilon_rate
        #
        # feature_mat = np.matrix(self.last_features)
        # delta_mat = np.matrix(self.last_deltas)

        deltas = self.deltas[:self.buffi]
        # print('delta : ' , self.deltas)
        #self.feature_buff = self.feature_buff[:self.buffi]
        feature_buff = self.feature_buff[:self.buffi]
        # print('feat: ', self.feature_buff)
        self.buffi = 0
        # print('w: ', self.weights)
        self.weights = self.weights - self.learning_rate * (deltas * feature_buff).mean(axis=0)

        #
        # # print(feature_mat.shape, delta_mat.shape, (delta_mat * feature_mat).shape)
        # # print(feature_mat, delta_mat, delta_mat.dot(feature_mat)/len(self.last_deltas))
        # #self.weights = self.weights - self.learning_rate * (delta_mat.dot(feature_mat))/len(self.last_deltas)
        # self.weights = self.weights - self.learning_rate * (delta_mat * feature_mat).mean(axis=0)

        self.weights = self.weights/self.weights.sum()

        # self.last_features = []
        # self.last_deltas = []

        try:
            if round % 100 == 0:
                if round > self.game_duration - self.score_scope:
                    self.log("Rewards in last 100 rounds which counts towards the score: " +
                             str(self.r_sum), 'VALUE')
                else:
                    self.log("Rewards in last 100 rounds: " + str(self.r_sum), 'VALUE')
                self.r_sum = 0
            else:
                self.r_sum += reward

        except Exception as e:
            self.log("Something Went Wrong...", 'EXCEPTION')
            self.log(e, 'EXCEPTION')

    def getQValue(self, state, action):
        board, head = state
        head_pos, direction = head
        # get a Position object of the position in the relevant direction from the head:
        next_position = head_pos.move(bp.Policy.TURNS[direction][action])
        r, c = next_position
        field = board[r, c]

        idx = field+1
        features = np.zeros(self.feature_num)
        features[idx] = 1
        q_value = self.weights[idx]  #[0, idx]
        return features, q_value

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        # if round == 1:
        #     print('this was first round')
        #     return np.random.choice(bp.Policy.ACTIONS)
        #
        if (round == 0) | (np.random.rand() < self.epsilon) & (round < self.game_duration - self.score_scope):
            # print('this was first round' + str(round))
            # action = np.random.choice(bp.Policy.ACTIONS)
            # q_max, features = self.getQValue(new_state, action)
            return np.random.choice(bp.Policy.ACTIONS)

        else:
            res = {'features': np.zeros([len(self.act2idx), self.feature_num]),
                   'q_values': np.zeros(3)}
            # for dir_idx, a in enumerate(list(np.random.permutation(bp.Policy.ACTIONS))):
            for a in self.act2idx:
                q_value, features_a = self.getQValue(new_state, a)
                res['features'][self.act2idx[a]] = features_a
                res['q_values'][self.act2idx[a]] = q_value

            q_max, q_max_idx = np.max(res['q_values']), np.argmax(res['q_values'])
            # features = res['features'][q_max_idx]
            action = self.idx2act[q_max_idx]

            # print(round, prev_action)
            prev_qvalue, prev_features = self.getQValue(prev_state, prev_action)

            # delta = prev_qvalue - (reward + (self.gamma * q_max))
            # self.last_features.append(prev_features) # is this it?
            # self.last_deltas.append(delta)
            #
            delta = prev_qvalue - (reward + (self.gamma * q_max))
            self.deltas[self.buffi] = delta
            self.feature_buff[self.buffi] = prev_features
            self.buffi += 1
            return action


class Linear(bp.Policy):
    """
    A policy which avoids collisions with obstacles and other snakes. It has an epsilon parameter which controls the
    percentag of actions which are randomly chosen.
    """

    def cast_string_args(self, policy_args):
        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
        policy_args['gamma'] = float(policy_args['gamma']) if 'gamma' in policy_args else GAMMA
        policy_args['learning_rate'] = float(policy_args['lr']) if 'lr' in policy_args else LEARNING_RATE

        return policy_args

    def init_run(self):
        self.r_sum = 0
        self.act2idx = {'L': 0, 'R': 1, 'F': 2}
        self.idx2act = {0: 'L', 1: 'R', 2: 'F'}
        self.deltas = np.zeros(shape=(LEARNING_TIME * 2, 1))
        self.weights = np.zeros(FEATURE_NUM)  # missin bias
        self.first_act = True
        self.feature_num = FEATURE_NUM
        self.last_features = []
        self.last_deltas = []
        self.epsilon_rate = EPSILON_RATE


    def getQValue(self, state, action):
        board, head = state
        head_pos, direction = head
        # get a Position object of the position in the relevant direction from the head:
        next_position = head_pos.move(bp.Policy.TURNS[direction][action])
        r, c = next_position
        field = board[r, c]

        idx = field+1
        features = np.zeros(self.feature_num)
        features[idx] = 1
        q_value = self.weights[idx]
        return features, q_value


    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):

        if self.epsilon >= 0.1:
            self.epsilon = self.epsilon * self.epsilon_rate

        delta_mat = np.matrix(self.last_deltas)
        feature_mat = np.matrix(self.last_features)

        weights = self.weights - self.learning_rate * (delta_mat.dot(feature_mat))/len(self.last_deltas)
        self.weights = np.array(weights.tolist()[0])

        self.last_deltas = []
        self.last_features = []


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

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):

        if round > self.game_duration - self.score_scope:
            self.epsilon = 0.0 # todo handle decreasing lr

        if not self.first_act:

            res = {'features': np.zeros([len(self.act2idx), self.feature_num]),
                   'q_values': np.zeros(3)}
            random_actions = np.random.permutation(bp.Policy.ACTIONS)
            for i, a in enumerate(random_actions):
                features_a, q_value = self.getQValue(new_state, a)
                res['features'][i] = features_a
                res['q_values'][i] = q_value

            q_max, a_idx = np.max(res['q_values']), np.argmax(res['q_values'])
            action = random_actions[a_idx]

            prev_features, prev_qvalue_f = self.getQValue(prev_state, prev_action)

            delta = prev_qvalue_f - (reward + (self.gamma * q_max))

            self.last_deltas.append(delta)
            self.last_features.append(prev_features)

        if np.random.rand() < self.epsilon or self.first_act:
            self.first_act = False
            return np.random.choice(bp.Policy.ACTIONS)

        return action
