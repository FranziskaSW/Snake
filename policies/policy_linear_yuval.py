from policies import base_policy as bp
import numpy as np

LEARNING_TIME = bp.LEARNING_TIME

EPSILON = 0.05
FEATURE_NUM = 11  # todo insert into dynamic func
GAMMA = 0.8
LEARNING_RATE = 0.01

class Linear_y(bp.Policy):
    """
    A linear Q-Learning approximation which avoids collisions with obstacles and other snakes ad 
    wish to eat good yummy fruits. It has an epsilon parameter which controls the
    percentag of actions which are randomly chosen.
    """

    def cast_string_args(self, policy_args):
        policy_args['epsilon'] = float(
            policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
        policy_args['gamma'] = float(
            policy_args['gamma']) if 'gamma' in policy_args else GAMMA
        policy_args['lr'] = float(
            policy_args['lr']) if 'lr' in policy_args else LEARNING_RATE
        return policy_args

    def init_run(self):
        self.r_sum = 0
        self.act2i = {'L': 0, 'R': 1, 'F': 2}
        self.i2act = {0: 'L', 1: 'R', 2: 'F'}
        self.buffi = 0
        self.feature_buff = np.zeros(shape=(LEARNING_TIME * 2, FEATURE_NUM))
        self.deltas = np.zeros(shape=(LEARNING_TIME * 2, 1))
        self.w = np.zeros(FEATURE_NUM)  # missin bias
        self.first_act = True

    def get_q(self, state, action):
        board, head = state
        head_pos, direction = head

        next_position = head_pos.move(bp.Policy.TURNS[direction][action])
        r = next_position[0]
        c = next_position[1]

        features = np.zeros(FEATURE_NUM)
        features[board[r, c] + 1] = 1

        return features, self.w[board[r, c] + 1]

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        self.deltas = self.deltas[:self.buffi]
        print('delta : ' , self.deltas)
        self.feature_buff = self.feature_buff[:self.buffi]
        print('feat: ', self.feature_buff)
        self.buffi = 0
        print('w: ', self.w)
        self.w = self.w - self.lr * (self.deltas * self.feature_buff).mean(axis=0)


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
        board, head = new_state
        head_pos, direction = head

        if round > self.game_duration - self.score_scope:
            self.epsilon = 0.0 # todo handle decreasing lr

        if not self.first_act:
            prev_features, prev_q = self.get_q(prev_state, prev_action)
            self.feature_buff[self.buffi] = prev_features

            # calculate delta and best act:
            qs = []
            acts = list(np.random.permutation(bp.Policy.ACTIONS))
            for a in acts:
                features, q = self.get_q(new_state, a)
                qs.append(q)
            best_act = acts[np.argmax(qs)]
            best_q = np.max(qs)

            delta = prev_q - (reward + (self.gamma * best_q))
            self.deltas[self.buffi] = delta
            self.buffi += 1

        if np.random.rand() < self.epsilon or self.first_act:
            self.first_act = False
            return np.random.choice(bp.Policy.ACTIONS)

        return best_act
