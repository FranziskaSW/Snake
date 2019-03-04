from policies import base_policy as bp
import math
from keras.models import Sequential
from keras.layers import *
import keras


EPSILON = 0.2
EPSILON_RATE = 0.9999
LEARNING_RATE = 0.001
BATCH_SIZE = 32
GAMMA = 0.5

NUM_ACTIONS = 3  # (L, R, F)
VICINITY = 3
DROPOUT_RATE = 0.2


class Custom777934738(bp.Policy):
    """
    A policy which uses a neural network to learn the dependency between the state representation and the reward.
    Calculates the q-value for the possible actions and acts according to the best q-value.
    Updates the Q-function via training of the net in order to converge towards the optimal Q-function, Bellman's equation.
    Has an epsilon parameter which controls the percentage of actions which are randomly chosen, which decreases over time.
    """

    def cast_string_args(self, policy_args):
        policy_args['epsilon'] = float(policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
        policy_args['epsilon_rate'] = float(policy_args['epsilon_rate']) \
            if 'epsilon_rate' in policy_args else EPSILON_RATE
        policy_args['batch_size'] = float(policy_args['batch_size']) if 'batch_size' in policy_args else BATCH_SIZE
        policy_args['vicinity'] = float(policy_args['vicinity']) if 'vicinity' in policy_args else VICINITY
        policy_args['learning_rate'] = float(policy_args['learning_rate']) \
            if 'learning_rate' in policy_args else LEARNING_RATE
        policy_args['gamma'] = float(policy_args['gamma']) if 'gamma' in policy_args else GAMMA
        return policy_args

    def init_run(self):
        self.r_sum = 0
        self.vicinity = VICINITY
        self.feature_num = (self.vicinity*2+1)**2*11 +1
        self.section_indices = np.array(range((self.vicinity*2+1)**2)) * 11
        self.input_shape = (self.feature_num, )
        self.memory = []
        self.act2idx = {'L': 0, 'R': 1, 'F': 2}
        self.idx2act = {0: 'L', 1: 'R', 2: 'F'}
        self.memory_length = int(self.batch_size*20)
        self.epsilon_rate = EPSILON_RATE
        self.not_too_slow_count = 0
        self.bs_init = self.batch_size

        # create the network
        self.model = Sequential()
        self.model.add(Dense(64, activation='relu', input_shape=self.input_shape))
        self.model.add(Dropout(DROPOUT_RATE))
        self.model.add(Dense(1))
        adam = keras.optimizers.Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        self.model.compile(loss='mean_squared_error', optimizer=adam)


    def train_model(self, batches):
        """
        update of the Q-function via training of the net
        Uses a random sample of transition tupels (state, action, reward, nest_state) and the current net
        to predict the targets y from the currrent state (called x).
        Then uses those x and y to train the model - update the Q-function.
        """
        bs_int = int(self.batch_size)
        x = np.zeros([bs_int, self.feature_num])
        y = np.zeros(bs_int)
        for idx, batch in enumerate(batches):
            x[idx] = batch['s_t']
            q_hat = (batch['r_t'] + self.gamma * np.max(self.model.predict(batch['s_tp1'])))
            y[idx] = q_hat

        self.model.fit(x, y, batch_size=bs_int, epochs=1, verbose=0)


    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        """
        Function for learning and improving the policy.
        If the memory replay is full enough, choose a random sample with size batch_size and train the network with it.
        Decrease exploration parameter epsilon by epsilon_rate but only as long as it is still bigger than 0.1
        (parameters as described in base_policy)
        """

        if self.epsilon >= 0.1:
            self.epsilon = self.epsilon * self.epsilon_rate

        if round >= self.batch_size:
            bs_int = int(self.batch_size)
            random_batches = np.random.choice(self.memory, bs_int)
            self.train_model(random_batches)

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
        """
        Creates a vicinity map around the center. Takes into account that the snake can move through the walls. Also
        turns the map so that the snake is looking to the top ('N')
        :param board: board of this round
        :param center: what is going to be the center of the vicinity map (usually next_position of head)
        :param direction: direction that the head looks at on board, so that the vicinity map can be turned that
                          head looks to north.
        :return: map of size (self.vicinity*2+1)**2 (7*7) with next_position in the center and snake looking to the top
        """
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
        """
        Translates the VicinityMap to the representation for the input of the network.
        Each pixel of the map is going to be translated to a 11-dimensional unit vector with 1 at the component
        of the field value, 0 otherwise. combines the 7*7=49 11-dim vectors to a 539-dim vector
        :param VicinityMap:
        :return: VicinityMap representation as 539-dim binary vector
        """
        symbols = np.matrix(VicinityMap.flatten())
        symbols = symbols[0]
        features = np.zeros(symbols.shape[1]*11)

        section_indices = self.section_indices
        symbol_idx = symbols + section_indices + 1
        symbol_idx_list = symbol_idx.tolist()[0]
        idx_int = [int(x) for x in symbol_idx_list]
        features[idx_int] = 1
        return features

    def getFeatures(self, board, head, action):
        """
        gets the features of the snake. Features: VicinityMap and length of snake
        :param board: board of current round
        :param head: current head of snake (position and direction)
        :param action: action that current head does to reach next_position
        :return: 540-dim vector, input of net
        """
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
        """
        Function that chooses an action given a current state.
        Performs the possible actions on the current state, calculates the q-value according to the features and chooses
        the action with the biggest q-value.
        Unless in case of exploration, in epsilon-% of the cases. Then choose random action.
        Updates the memory replay with previous transition tuple (prev_features, prev_action, reward, new_features)
        Also decreases the batch_size if agent is too slow.
        (parameters as described in base_policy)
        """

        if too_slow & (round >= 100):
            self.batch_size = math.ceil(self.batch_size/2)
            self.not_too_slow_count = 0
        else:
            self.not_too_slow_count += 1

        if (self.not_too_slow_count == 200) & (self.batch_size < self.bs_init):
            self.batch_size = int(self.batch_size*2)


        board, head = new_state
        new_features = np.zeros([len(self.act2idx), self.feature_num])

        random_actions = np.random.permutation(bp.Policy.ACTIONS)
        for i, a in enumerate(random_actions):
            new_features[i] = self.getFeatures(board, head, a)

        if round >=2:  # update to memory from previous round (prev_state)
            prev_board, prev_head = prev_state
            prev_feature = self.getFeatures(prev_board, prev_head, prev_action)
            memory_update = {'s_t': prev_feature, 'a_t': self.act2idx[prev_action], 'r_t': reward, 's_tp1': new_features}

            if len(self.memory) < self.memory_length:
                self.memory.append(memory_update)
            else:
                idx = np.random.choice(range(0, self.memory_length))
                self.memory[idx] = memory_update

        # act in new round, decide for new_state
        if (np.random.rand() < self.epsilon) & (round < self.game_duration - self.score_scope):
            action = np.random.choice(bp.Policy.ACTIONS)

        else:
            q_values = self.model.predict(new_features, batch_size=3)
            a_idx = np.argmax(q_values)
            action = random_actions[a_idx]

        return action
