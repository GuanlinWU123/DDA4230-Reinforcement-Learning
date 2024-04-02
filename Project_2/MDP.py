import numpy as np


class MDP:
    '''A simple MDP class.  It includes the following members'''

    def __init__(self, T, R, discount):
        '''Constructor for the MDP class

        Inputs:
        T -- Transition function: |A| x |S| x |S'| array
        R -- Reward function: |A| x |S| array
        discount -- discount factor: scalar in [0,1)

        The constructor verifies that the inputs are valid and sets
        corresponding variables in a MDP object'''

        assert T.ndim == 3, "Invalid transition function: it should have 3 dimensions"
        self.nActions = T.shape[0]
        self.nStates = T.shape[1]
        assert T.shape == (
            self.nActions, self.nStates, self.nStates), "Invalid transition function: it has dimensionality " + repr(
            T.shape) + ", but it should be (nActions,nStates,nStates)"
        assert (abs(T.sum(
            2) - 1) < 1e-5).all(), "Invalid transition function: some transition probability does not equal 1"
        self.T = T
        assert R.ndim == 2, "Invalid reward function: it should have 2 dimensions"
        assert R.shape == (self.nActions, self.nStates), "Invalid reward function: it has dimensionality " + repr(
            R.shape) + ", but it should be (nActions,nStates)"
        self.R = R
        assert 0 <= discount < 1, "Invalid discount factor: it should be in [0,1)"
        self.discount = discount

    def valueIteration(self, initialV, nIterations=np.inf, tolerance=0.01):
        V = initialV.copy()
        iterId = 0
        epsilon = 0
        while iterId < nIterations:
            iterId += 1
            V_values = self.R + self.discount * np.dot(self.T, V)
            V_new = np.max(self.R + self.discount * np.dot(self.T, V), axis=0)
            epsilon = np.max(abs(V_new - V))
            V = V_new
            if epsilon < tolerance:
                break
            # print("Value Iteration: \nV = ", V, "\niterId = ", iterId, "\nepsilon = ", epsilon)
        # print("Q_values: ", Q_values)
        # print("V_new: ",V_new)
        policy = np.argmax(V_values, axis=0)
        print("policy:", policy)
        return [V, iterId, epsilon]

    def extractPolicy(self, V):
        Q = self.R + self.discount * np.dot(self.T, V)
        policy = np.argmax(Q, axis=0)
        return policy

    def evaluatePolicy(self, policy):
        R_policy = np.array([self.R[policy[i], i] for i in range(self.nStates)])
        T_policy = np.array([self.T[policy[i], i, :] for i in range(self.nStates)])
        I = np.identity(self.nStates)
        V = np.linalg.solve(I - self.discount * T_policy, R_policy)
        return V

    def policyIteration(self, initialPolicy, nIterations=np.inf):
        policy = initialPolicy.copy()
        V = np.zeros(self.nStates)
        iterId = 0
        while iterId < nIterations:
            iterId += 1
            V = self.evaluatePolicy(policy)
            new_policy = self.extractPolicy(V)
            if (policy == new_policy).all():
                break
            policy = new_policy.copy()
        return [policy, V, iterId]

    def evaluatePolicyPartially(self, policy, initialV, nIterations=np.inf, tolerance=0.01):
        V = initialV.copy()
        R_policy = np.array([self.R[policy[i], i] for i in range(self.nStates)])
        T_policy = np.array([self.T[policy[i], i, :] for i in range(self.nStates)])
        iterId = 0
        epsilon = 0
        while iterId < nIterations:
            iterId += 1
            V_new = R_policy + self.discount * np.dot(T_policy, V)
            epsilon = np.max(abs(V_new - V))
            V = V_new
            if epsilon < tolerance:
                break
        return [V, iterId, epsilon]

    def modifiedPolicyIteration(self, initialPolicy, initialV, nEvalIterations=5, nIterations=np.inf, tolerance=0.01):
        policy = initialPolicy.copy()
        V = initialV.copy()
        iterId = 0
        epsilon = 0
        while iterId < nIterations:
            iterId += 1
            V, _, epsilon = self.evaluatePolicyPartially(policy, V, nEvalIterations, tolerance)
            new_policy = self.extractPolicy(V)
            if (policy == new_policy).all():
                break
            policy = new_policy.copy()
        return [policy, V, iterId, epsilon]

