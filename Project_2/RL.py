import numpy as np
import MDP

class RL:
    def __init__(self,mdp,sampleReward):
        '''Constructor for the RL class

        Inputs:
        mdp -- Markov decision process (T, R, discount)
        sampleReward -- Function to sample rewards (e.g., bernoulli, Gaussian).
        This function takes one argument: the mean of the distributon and 
        returns a sample from the distribution.
        '''

        self.mdp = mdp
        self.sampleReward = sampleReward

    def sampleRewardAndNextState(self,state,action):
        '''Procedure to sample a reward and the next state
        reward ~ Pr(r)
        nextState ~ Pr(s'|s,a)

        Inputs:
        state -- current state
        action -- action to be executed

        Outputs: 
        reward -- sampled reward
        nextState -- sampled next state
        '''

        reward = self.sampleReward(self.mdp.R[action,state])
        cumProb = np.cumsum(self.mdp.T[action,state,:])
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward,nextState]

    def qLearning(self,s0,initialQ,nEpisodes,nSteps,epsilon=0,temperature=0):
        '''qLearning algorithm.  
        When epsilon > 0: perform epsilon exploration (i.e., with probability epsilon, select action at random )
        When epsilon == 0 and temperature > 0: perform Boltzmann exploration with temperature parameter
        When epsilon == 0 and temperature == 0: no exploration (i.e., selection action with best Q-value)

        Inputs:
        s0 -- initial state
        initialQ -- initial Q function (|A|x|S| array)
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- # of steps per episode
        epsilon -- probability with which an action is chosen at random
        temperature -- parameter that regulates Boltzmann exploration

        Outputs: 
        Q -- final Q function (|A|x|S| array)
        policy -- final policy
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        # Q = np.zeros([self.mdp.nActions,self.mdp.nStates])
        # policy = np.zeros(self.mdp.nStates,int)

        Q = initialQ.copy()
        alpha = 0.1
        for episode in range(nEpisodes):
            state = s0
            for step in range(nSteps):
                # Epsilon-greedy exploration
                if epsilon > 0 and np.random.rand() < epsilon:
                    action = np.random.choice(self.mdp.nActions)
                # Boltzmann exploration
                elif epsilon == 0 and temperature > 0:
                    probabilities = np.exp(Q[:, state] / temperature)
                    probabilities /= probabilities.sum()
                    action = np.random.choice(self.mdp.nActions, p=probabilities)
                # Greedy action selection
                else:
                    action = np.argmax(Q[:, state])

                # Sample reward and next state
                reward, nextState = self.sampleRewardAndNextState(state, action)

                # Q-learning update
                Q[action, state] += alpha * (reward + self.mdp.discount * np.max(Q[:, nextState]) - Q[action, state])

                # Move to the next state
                state = nextState

        # Extract policy from Q-values
        policy = np.argmax(Q, axis=0)
        return [Q, policy]

        # return [Q,policy]

    def compute_average_cumulative_reward(self, s0, initialQ, nEpisodes, nSteps, epsilon, temperature=0, nTrials=100):
        total_rewards = np.zeros(nEpisodes)

        for trial in range(nTrials):
            Q = initialQ.copy()
            cumulative_reward = 0
            discount_factor = 1
            for episode in range(nEpisodes):
                _, nextState = self.sampleRewardAndNextState(s0, np.argmax(Q[:, s0]))
                reward = self.sampleReward(self.mdp.R[np.argmax(Q[:, s0]), s0])
                cumulative_reward += discount_factor * reward
                discount_factor *= self.mdp.discount
                Q, _ = self.qLearning(s0, Q, 1, nSteps, epsilon, temperature)  # Running one episode at a time
                total_rewards[episode] += cumulative_reward

        average_rewards = total_rewards / nTrials
        return average_rewards