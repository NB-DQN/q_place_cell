import random

class QAgent:
    def __init__(self, environment):
        self.environment = environment
        self.q = {}

    def choose_action(self):
        EPSILON = 0.1
        s = self.environment.coordinate_id()

        if random.random() < EPSILON:
            a = random.choice(self.get_available_actions(s))
        else:
            a = self.choose_action_greedy(s)
        self.environment.move(a)
        s_next = self.environment.coordinate_id()
        self.update_qsa(s, a, self.environment.coordinate_id())
        return s, a, s_next

    def choose_action_greedy(self, s):
        bests = []
        qsa_max = 0
        for a in self.get_available_actions(s):
            qsa = self.get_qsa(s, a)
            if qsa > qsa_max:
                bests = [a]
                qsa_max = qsa
            elif qsa == qsa_max:
                bests.append(a)
        return random.choice(bests)

    def update_qsa(self, s, a, s_new):
        ALPHA = 0.5
        GAMMA = 0.5
        qsa = self.get_qsa(s, a)
        qsa_max = max([self.get_qsa(s_new, a_new) for a_new in self.get_available_actions(s_new)])
        rsa = 10 if self.environment.get_goal() else 0

        qsa_new = qsa + ALPHA * (rsa + GAMMA * qsa_max - qsa)
        self.q.setdefault(s, {})
        self.q[s][a] = qsa_new

    def get_qsa(self, s, a):
        try:
            return self.q[s][a]
        except KeyError:
            return 0

    def get_available_actions(self, s):
        wall = self.environment.wall(s)
        actions = []
        for a in range(0, 4):
            if wall[a] == 0:
                actions.append(a)
        return actions

    def dump_q(self):
        pass
