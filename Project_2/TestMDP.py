from MDP import *

''' Construct simple MDP as described in Figure 2'''
# Transition function: |A| x |S| x |S'| array
T = np.array([[[0.5,0.5,0,0],[0,1,0,0],[0.5,0.5,0,0],[0,1,0,0]],[[1,0,0,0],[0.5,0,0,0.5],[0.5,0,0.5,0],[0,0,0.5,0.5]]])
# Reward function: |A| x |S| array
R = np.array([[0,0,10,10],[0,0,10,10]])
# Discount factor: scalar in [0,1)
discount = 0.9        
# MDP object
mdp = MDP(T,R,discount)

'''Test each procedure'''
[V,nIterations,epsilon] = mdp.valueIteration(initialV=np.zeros(mdp.nStates))
print("-------------Value-------------")
print("Value Iteration: \nV = ", V, "\niterId = ", nIterations, "\nepsilon = ", epsilon)

print("-------------Policy-------------")
policy = mdp.extractPolicy(V=np.zeros(mdp.nStates))
print("policy: ", policy)
V = mdp.evaluatePolicy(np.array([1,0,1,0]))
print("V: ", V)
[policy,V,iterId] = mdp.policyIteration(np.array([0,0,0,0]))
print("Policy Iteration: \npolicy = ", policy, "\nV: ", V, "\niterId = ", iterId)

print("-------------Policy Modified-------------")
[V,iterId,epsilon] = mdp.evaluatePolicyPartially(np.array([1,0,1,0]),np.array([0,10,0,13]))
print("Evaluate Policy Partially: \nV: ", V, "\niterId = ", iterId, "\nepsilon = ", epsilon)
[policy,V,iterId,epsilon] = mdp.modifiedPolicyIteration(np.array([1,0,1,0]),np.array([0,10,0,13]))
print("Modified Policy Iteration: \npolicy = ", policy, "\nV: ", V, "\niterId = ", iterId,
      "\nepsilon: ", epsilon)
