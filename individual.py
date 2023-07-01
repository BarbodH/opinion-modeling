from random import choice


class Individual:
    def __init__(self, network_index):
        # disease status
        # 0: susceptible (S)
        # 1: asymptomatic infection (I_A)
        # 2: symptomatic infection (I_S)
        # 3: recovered (R)
        self._disease = 0
        # opinion toward quarantine
        # -1: against
        # 1: for
        self._opinion = choice([-1, 1])
        # quarantine status
        # 0: does not adhere to lockdown
        # 1: adheres to lockdown
        self._quarantine = 1 if self.opinion == 1 else 0
        # timer to count days of sickness
        self._self_timer = 0

        self._network_index = network_index

    # accessor methdos

    @property
    def disease(self):
        return self._disease

    @property
    def opinion(self):
        return self._opinion

    @property
    def self_timer(self):
        return self._self_timer

    @property
    def quarantine(self):
        return self._quarantine

    @property
    def network_index(self):
        return self._network_index

    # modifier methods

    @disease.setter
    def disease(self, value: int):
        if value >= 0 and value <= 3:
            self._disease = value

    @opinion.setter
    def opinion(self, value: int):
        if value == 1 or value == -1:
            self._opinion = value
            self._update_quarantine()

    @quarantine.setter
    def quarantine(self, value: int):
        if value == 0 or value == 1:
            self._quarantine = value

    # utility methods

    def _update_quarantine(self):
        self._quarantine = 1 if self.opinion == 1 else 0

    def increment_self_timer(self):
        self._self_timer += 1

    def recover(self):
        self._disease = 3

    def get_infected(self):
        self._disease = 1
        self._self_timer = 0

    def show_symptoms(self):
        self._disease = 2
        self._self_timer = 0
        self._opinion = 2
        self._quarantine = 1
