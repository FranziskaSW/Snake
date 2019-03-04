from policies import base_policy as bp
import numpy as np

EPSILON = 0.2
EPSILON_RATE = 0.9999
GAMMA = 0.5
LEARNING_RATE = 0.01
FEATURE_NUM = 11
LEARNING_TIME = bp.LEARNING_TIME

FEATURE_NUM = 11


class Linear777934738(bp.Policy):
    """
    A policy which learns a linear dependency between the value of the next field and the reward. Calculates the q-value
    for the possible actions and acts according to the best q-value.
    It has an epsilon parameter which controls the percentage of actions which are randomly chosen.
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
        self.weights = np.zeros(FEATURE_NUM)
        self.action1 = True
        self.feature_num = FEATURE_NUM
        self.last_features = []
        self.last_deltas = []
        self.epsilon_rate = EPSILON_RATE

    def getQValue(self, state, action):
        """
        Creates the features of the next state and calculates the q-value accordingly.
        The feature is a 11-dimensional unit vector with 1 at the component of the field value of the next state,
        0 otherwise.
        :param state: the current state (board, head position, head direction)
        :param action: the action that is going to be performed
        :return: feature-vector of the next state, q-value of the next state
        """
        board, head = state
        head_pos, direction = head
        # move the snake to next position
        next_position = head_pos.move(bp.Policy.TURNS[direction][action])
        r, c = next_position
        field = board[r, c]

        idx = field+1
        features = np.zeros(self.feature_num)
        features[idx] = 1
        q_value = self.weights[idx]
        return features, q_value


    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        """
        Function for learning and improving the policy.
        In the linear approximation of the Q-function the gradient of the Q-function is just the feature vector.
        Decrease exploration parameter epsilon by epsilon_rate but only as long as it is still bigger than 0.1
        (parameters as described in base_policy)
        """

        if self.epsilon >= 0.1:
            self.epsilon = self.epsilon * self.epsilon_rate

        # get deltas and features of the last action rounds that were performed after last training
        delta_mat = np.matrix(self.last_deltas)
        feature_mat = np.matrix(self.last_features)

        # update the Q-function equal to update the weights since we have linear approx. of Q-function
        weights = self.weights - self.learning_rate * (delta_mat.dot(feature_mat))/len(self.last_deltas)
        self.weights = np.array(weights.tolist()[0])

        # empty the list of last deltas and features
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
        """
        Function that chooses an action given a current state.
        Performs the possible actions on the current state, calculates the q-value according to the features and chooses
        the action with the biggest q-value.
        Unless in case of exploration, in epsilon-% of the cases. Then choose random action.
        Updates the short term memory with previous values (delta, features)
        (parameters as described in base_policy)
        """

        if round > self.game_duration - self.score_scope:
            self.epsilon = 0.0

        if not self.action1:
            # choose best action according to maximum q-value
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

            # calculate delta, which is needed for updating of the Q-function in the learning step
            delta = prev_qvalue_f - (reward + (self.gamma * q_max))

            self.last_deltas.append(delta)
            self.last_features.append(prev_features)

        if np.random.rand() < self.epsilon or self.action1:
            # if it is the first action, it does not make sense to calculate a q-value because it is not based on anything
            # also we can not update the short term memory with the previous features because there are none
            self.action1 = False
            return np.random.choice(bp.Policy.ACTIONS)

        return action
