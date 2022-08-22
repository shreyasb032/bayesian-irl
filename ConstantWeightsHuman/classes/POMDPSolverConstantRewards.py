import numpy as np
from copy import copy

class Solver:
    
    def __init__(self, N, wh, wc, wt, params_list, prior_levels, after_scan_levels, threats, est_human_weights, hl=5, tc=10, df=0.9, use_constant=False):

        # Number of stages
        self.N = N

        # Health reward
        self.wh = wh
        self.hl = hl
        self.use_constant = use_constant

        # Time reward
        self.wc = wc
        self.tc = tc

        # Trust reward
        self.wt = wt

        # Discount factor
        self.df = df
        
        # Estimated human reward weights
        self.est_human_weights = est_human_weights

        # Storage
        self.performance_history = []
        self.threat_levels = copy(prior_levels)
        self.params_list = copy(params_list)
        self.after_scan_levels = copy(after_scan_levels)
        self.threats = threats
        self.max_health = 100
        self.health = 100

    def reset(self):
        self.performance_history = []

    def set_est_human_weights(self, est_human_weights):
        self.est_human_weights = est_human_weights

    def set_reward_weights(self, wh, wc):

        self.wh = wh
        self.wc = wc
        self.reset()
        
    def get_immediate_reward(self, current_house, current_health, action):
        
        r_follow = 0
        r_not_follow = 0

        hl = self.hl
        
        if not self.use_constant:
            hl = self.hl / (current_health - 49)

        r1 = -self.wc * self.tc
        r2 = -self.wh * hl
        r3 = 0

        if action == 1:
            r_follow = r1
            if self.threats[current_house] == 1:
                r_not_follow = r2
            else:
                r_not_follow = r3
        else:
            r_not_follow = r1
            if self.threats[current_house] == 1:
                r_follow = r2
            else:
                r_follow = r3

        return r_follow, r_not_follow

    def health_loss_reward(self, health):
        # Linear
        return health - self.max_health

    def time_loss_reward(self, curr_time):
        # Linear
        return -curr_time

    def get_values(self, current_house, current_health, params, after_scan):
        
        alpha_0 = params[0]
        beta_0 = params[1]
        ws = params[2]
        wf = params[3]

        ns = np.sum(self.performance_history)
        nf = current_house - ns

        alpha_previous = alpha_0 + ws * ns
        beta_previous = beta_0 + wf * nf

        if current_health < 50:
            current_health = 50

        if not self.use_constant:
            self.hl = 200 / (current_health - 49)

        ryy = -self.wc * self.tc
        ryn = -self.wh * self.hl
        rny = -self.wc * self.tc
        rnn = 0

        i = current_house
        n = self.N

        value_matrix = np.zeros((n - i + 1, n - i + 1), dtype=float)         # Extra stage of value zero
        self.threat_levels[i] = after_scan

        for jj in range(i, n):
            #jj goes from i to n-1
            
            j = n + i - jj
            #j goes from n to i + 1

            ps = np.arange(j - i)      # Possible number of successes (Should not count the last one as the ending trust value is not getting used)
            pf = j - i - ps            # Possible number of failures (Corresponding)

            alpha = alpha_previous + ps * ws     # Possible values of alpha
            beta = beta_previous + pf * wf       # Possible values of beta

            threat = self.threat_levels[j-1]     # Threat level at current site

            trust_weight = self.wt
            gamma = self.df
            
            trust = alpha / (alpha + beta)

            # Value for recommending to use RARV
            reward_at_current_site = trust * (threat * (ryy-rny) + rny) + (1 - trust) * (threat * (ryn-rnn) + rnn)
            trust_gain_reward = trust_weight * threat * np.sqrt(jj)
            
            rew2use = -self.est_human_weights["time"]
            rew2notuse = -self.est_human_weights["health"]

            # TODO: This needs to be changed. Trust increases if immediate reward for human to follow > to not follow
            # Here recommendation is to use RARV. We can compute expected immediate reward for the human and update trust based on that
            # prob_performance_1 = p(reward2follow > reward2notfollow) = p(reward to use RARV > reward to not use RARV)
            # reward to use RARV = -wc_est
            # reward to not use RARV = -p(threat) * wh_est
            # value_at_next_stage = (threat * value_matrix[1:j-i+1, j-i] + (1-threat) * value_matrix[0:j-i, j-i])
            fac = int(rew2use >= rew2notuse)
            value_at_next_stage = fac * value_matrix[1:j-i+1, j-i] + (1-fac) * value_matrix[0:j-i, j-i]

            Vjy = reward_at_current_site + trust_gain_reward + gamma*value_at_next_stage

            # Value for recommending to not use RARV
            reward_at_current_site = trust * (threat * (ryn-rnn) + rnn) + (1-trust) * (threat * (ryy-rny) + rny)
            trust_gain_reward = trust_weight * (1-threat) * np.sqrt(jj)

            # TODO: This needs to be changed. Trust increases if immediate reward for human to follow > to not follow
            # Here recommendation is to not use RARV. We can compute expected immediate reward for the human and update trust based on that
            # prob_performance_1 = p(reward2follow > reward2notfollow) = p(reward to not use RARV > reward to use RARV)
            # reward to use RARV = -wc_est
            # reward to not use RARV = -p(threat) * wh_est
            fac = int(rew2notuse >= rew2use)
            value_at_next_stage = (fac * value_matrix[1:j-i+1, j-i] + (1-fac) * value_matrix[0:j-i,j-i])

            Vjn = reward_at_current_site + trust_gain_reward + gamma * value_at_next_stage

            for k in range(j-i):
                if (Vjy[k] >= Vjn[k]):
                    value_matrix[k, j - i - 1] = Vjy[k]
                else:
                    value_matrix[k, j - i - 1] = Vjn[k]

        return Vjy[0], Vjn[0]

    def get_action(self, current_house, current_health, params, after_scan):

        alpha_0 = params[0]
        beta_0 = params[1]
        ws = params[2]
        wf = params[3]

        ns = np.sum(self.performance_history)
        nf = current_house - ns

        alpha_previous = alpha_0 + ws * ns
        beta_previous = beta_0 + wf * nf

        current_health = self.health
        
        if current_health < 50:
            current_health = 50

        if not self.use_constant:
            self.hl = 200 / (current_health - 49)
        
        ryy = -self.wc * self.tc
        ryn = -self.wh * self.hl
        rny = -self.wc * self.tc
        rnn = 0

        i = current_house
        n = self.N

        value_matrix = np.zeros((n - i + 1, n - i + 1), dtype=float)         # Extra stage of value zero
        action_matrix = np.zeros((n - i, n - i), dtype=int)

        self.threat_levels[i] = after_scan

        for jj in range(i, n):
            #jj goes from i to n-1

            j = n + i - jj
            #j goes from n to i+1

            ps = np.arange(j - i)      # Possible number of successes (Should not count the last one as the ending trust value is not getting used)
            pf = j - i - ps            # Possible number of failures (Corresponding)

            alpha = alpha_previous + ps * ws     # Possible values of alpha
            beta = beta_previous + pf * wf       # Possible values of beta

            threat = self.threat_levels[j-1]     # Threat level at current site

            trust_weight = self.wt
            gamma = self.df
            
            trust = alpha / (alpha + beta)

            # Value for recommending to use RARV
            reward_at_current_site = trust * (threat * (ryy-rny) + rny) + (1 - trust) * (threat * (ryn-rnn) + rnn)
            trust_gain_reward = trust_weight * threat * np.sqrt(jj)
            
            rew2use = -self.est_human_weights["time"]
            rew2notuse = -self.est_human_weights["health"] * threat

            # TODO: This needs to be changed. Trust increases if immediate reward for human to follow > to not follow
            # Here recommendation is to use RARV. We can compute expected immediate reward for the human and update trust based on that
            # prob_performance_1 = p(reward2follow > reward2notfollow) = p(reward to use RARV > reward to not use RARV)
            # reward to use RARV = -wc_est
            # reward to not use RARV = -p(threat) * wh_est
            # value_at_next_stage = (threat * value_matrix[1:j-i+1, j-i] + (1-threat) * value_matrix[0:j-i, j-i])
            fac = int(rew2use >= rew2notuse)
            value_at_next_stage = fac * value_matrix[1:j-i+1, j-i] + (1-fac) * value_matrix[0:j-i, j-i]

            Vjy = reward_at_current_site + trust_gain_reward + gamma*value_at_next_stage

            # Value for recommending to not use RARV
            reward_at_current_site = trust * (threat * (ryn-rnn) + rnn) + (1-trust) * (threat * (ryy-rny) + rny)
            trust_gain_reward = trust_weight * (1-threat) * np.sqrt(jj)

            # TODO: This needs to be changed. Trust increases if immediate reward for human to follow > to not follow
            # Here recommendation is to not use RARV. We can compute expected immediate reward for the human and update trust based on that
            # prob_performance_1 = p(reward2follow > reward2notfollow) = p(reward to not use RARV > reward to use RARV)
            # reward to use RARV = -wc_est
            # reward to not use RARV = -p(threat) * wh_est
            fac = int(rew2notuse >= rew2use)
            value_at_next_stage = (fac * value_matrix[1:j-i+1, j-i] + (1-fac) * value_matrix[0:j-i,j-i])

            Vjn = reward_at_current_site + trust_gain_reward + gamma * value_at_next_stage

            for k in range(j-i):
                if (Vjy[k] >= Vjn[k]):
                    value_matrix[k, j - i - 1] = Vjy[k]
                    action_matrix[k, j - i - 1] = 1
                else:
                    value_matrix[k, j - i - 1] = Vjn[k]
                    action_matrix[k, j - i - 1] = 0

        return action_matrix[0,0]

    def forward(self, current_house, rec):

        rew2use = -self.est_human_weights["time"]
        rew2notuse = -self.est_human_weights["health"] * self.threats[current_house]
        
        if rec:
            if rew2use >= rew2notuse:
                self.performance_history.append(1)
            else:
                self.performance_history.append(0)
        else:
            if rew2notuse >= rew2use:
                self.performance_history.append(1)
            else:
                self.performance_history.append(0)

    def get_last_performance(self):

        return self.performance_history[-1]

    def get_policy(self):

        policy = []
        self.health = 100

        # Get current action. Update according to current action. Append this to policy. Repeat for all houses.
        for i in range(100):
            action = self.get_action(i, self.health, self.params_list[i], self.after_scan_levels[i])
            policy.append(action)
            self.forward(i, action)
        
        return policy

    def evaluate_policy(self, policy, params):
        # Here policy gives a vector of actions to be taken at each time step.
        value = 0
        alpha_0 = params[0]
        beta_0 = params[1]
        ws = params[2]
        wf = params[1]
        ns = 0
        nf = 0

        for i in range(len(policy)):
            rec = policy[i]
            alpha = alpha_0 + ns*ws
            beta = beta_0 + nf*wf
            trust = alpha/(alpha+beta)
            
            if policy[i] == self.threats[i]:
                ns += 1
            else:
                nf += 1

            health_loss = self.hl * self.after_scan_levels[i] * (trust * (1-rec) + (1-trust)*rec)
            time_loss = self.tc * (trust * rec + (1-trust) * (1-rec))
            trust_gain = np.sqrt(len(policy) - i) * (self.after_scan_levels[i] * rec + (1- self.after_scan_levels[i]) * (1-rec))

            value = value - self.wh * health_loss - self.wc * time_loss + self.wt * trust_gain
        
        return value

    def get_trust_estimate(self, params):

        per = np.sum(self.performance_history)
        _alpha = params[0] + per * params[2]
        _beta = params[1] + (len(self.performance_history) - per) * params[3]
        
        return _alpha / (_alpha + _beta)
    

class StageSolver(Solver):
    
    def __init__(self, N, wh, wc, wt, params_list, prior_levels, after_scan_levels, threats, hl=200, tc=150, df=0.95, use_constant=False, look_ahead = 3, use_trust_reward=True):
        super().__init__(N, wh, wc, wt, params_list, prior_levels, after_scan_levels, threats, hl=hl, tc=tc, df=df, use_constant=use_constant)
        self.look_ahead = look_ahead
        self.use_trust_reward = use_trust_reward
    
    def get_action(self, current_house, current_health, params, after_scan):
        """Only solve the POMDP considering the horizon to be look_ahead number of houses"""
        
        alpha_0 = params[0]
        beta_0 = params[1]
        ws = params[2]
        wf = params[3]

        ns = np.sum(self.performance_history)
        nf = current_house - ns

        alpha_previous = alpha_0 + ws * ns
        beta_previous = beta_0 + wf * nf

        if current_health < 50:
            current_health = 50

        if not self.use_constant:
            self.hl = 200 / (current_health - 49)
        
        ryy = -self.wc * self.tc
        ryn = -self.wh * self.hl
        rny = -self.wc * self.tc
        rnn = 0        

        i = current_house
        n = min(self.N, i + self.look_ahead)               # The minimum ensures that we do not go beyond the total number of houses, while only looking at look_ahead number of sites ahead

        value_matrix = np.zeros((n - i + 1, n - i + 1), dtype=float)         # Extra stage of value zero
        action_matrix = np.zeros((n - i, n - i), dtype=int)

        self.threat_levels[i] = after_scan

        for jj in range(i, n):
            #jj goes from i to n-1
            
            j = n + i - jj
            #j goes from n to i + 1

            ps = np.arange(j - i)      # Possible number of successes (Should not count the last one as the ending trust value is not getting used)
            pf = j - i - ps            # Possible number of failures (Corresponding)

            alpha = alpha_previous + ps * ws     # Possible values of alpha
            beta = beta_previous + pf * wf       # Possible values of beta

            threat = self.threat_levels[j-1]     # Threat level at current site

            trust_weight = self.wt
            gamma = self.df
            
            trust = alpha / (alpha + beta)

            # Value for recommending to use RARV
            reward_at_current_site = trust*(threat*(ryy-rny)+rny) + (1 - trust)*(threat*(ryn-rnn)+rnn)
            trust_gain_reward = trust_weight*threat*np.sqrt(jj)
            value_at_next_stage = (threat*value_matrix[1:j-i+1, j-i] + (1-threat)*value_matrix[0:j-i, j-i])

            Vjy = reward_at_current_site + trust_gain_reward + gamma*value_at_next_stage

            # Value for recommending to not use RARV
            reward_at_current_site = trust*(threat*(ryn-rnn)+rnn) + (1-trust)*(threat*(ryy-rny)+rny)
            trust_gain_reward = trust_weight*(1-threat)*np.sqrt(jj)
            value_at_next_stage = ((1-threat)*value_matrix[1:j-i+1, j-i] + threat * value_matrix[0:j-i,j-i])

            Vjn = reward_at_current_site + trust_gain_reward + gamma*value_at_next_stage

            for k in range(j-i-1):
                if (Vjy[k] >= Vjn[k]):
                    value_matrix[k, j - i - 1] = Vjy[k]
                    action_matrix[k, j - i - 1] = 1
                else:
                    value_matrix[k, j - i - 1] = Vjn[k]
                    action_matrix[k, j - i - 1] = 0

        return action_matrix[0,0]
    
    def get_values(self, current_house, current_health, params, after_scan):
        """Only solve the POMDP considering the horizon to be look_ahead number of houses"""

        alpha_0 = params[0]
        beta_0 = params[1]
        ws = params[2]
        wf = params[3]

        ns = np.sum(self.performance_history)
        nf = current_house - ns

        alpha_previous = alpha_0 + ws * ns
        beta_previous = beta_0 + wf * nf

        if current_health < 50:
            current_health = 50

        if not self.use_constant:
            self.hl = 200 / (current_health - 49)

        ryy = -self.wc * self.tc
        ryn = -self.wh * self.hl
        rny = -self.wc * self.tc
        rnn = 0

        i = current_house
        n = min(self.N, i + self.look_ahead)               # The minimum ensures that we do not go beyond the total number of houses, while only looking at look_ahead number of sites ahead

        value_matrix = np.zeros((n - i + 1, n - i + 1), dtype=float)         # Extra stage of value zero
        self.threat_levels[i] = after_scan

        for jj in range(i, n):
            #jj goes from i to n-1

            j = n + i - jj
            #j goes from n to i + 1

            ps = np.arange(j - i)      # Possible number of successes (Should not count the last one as the ending trust value is not getting used)
            pf = j - i - ps            # Possible number of failures (Corresponding)

            alpha = alpha_previous + ps * ws     # Possible values of alpha
            beta = beta_previous + pf * wf       # Possible values of beta

            threat = self.threat_levels[j-1]     # Threat level at current site

            trust_weight = self.wt
            gamma = self.df
            
            trust = alpha / (alpha + beta)

            # Value for recommending to use RARV
            reward_at_current_site = trust*(threat*(ryy-rny)+rny) + (1 - trust)*(threat*(ryn-rnn)+rnn)
            value_at_next_stage = (threat*value_matrix[1:j-i+1, j-i] + (1-threat)*value_matrix[0:j-i, j-i])

            Vjy = reward_at_current_site + gamma*value_at_next_stage
            
            if self.use_trust_reward:
                trust_gain_reward = trust_weight*threat*np.sqrt(jj)
                Vjy += trust_gain_reward

            # Value for recommending to not use RARV
            reward_at_current_site = trust*(threat*(ryn-rnn)+rnn) + (1-trust)*(threat*(ryy-rny)+rny)
            value_at_next_stage = ((1-threat)*value_matrix[1:j-i+1, j-i] + threat * value_matrix[0:j-i,j-i])

            Vjn = reward_at_current_site + gamma*value_at_next_stage

            if self.use_trust_reward:
                trust_gain_reward = trust_weight*(1-threat)*np.sqrt(jj)
                Vjn += trust_gain_reward

            for k in range(j-i):
                if (Vjy[k] >= Vjn[k]):
                    value_matrix[k, j - i - 1] = Vjy[k]
                else:
                    value_matrix[k, j - i - 1] = Vjn[k]

        return Vjy[0], Vjn[0]

