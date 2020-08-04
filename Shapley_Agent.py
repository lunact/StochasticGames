from non_RL_Agent import non_RL_Agent
import numpy as np
import random
import copy
import StaticGameResolution as sgr

def new_V_and_pi(game, V):
    """
    compute the next value of V based on its current value
    
    :param game: NullSum2PlayerStochasticGame
    :param V: current value fonction (dictionary: {state: float})
    :rtype: player 0 strategy (dictionary: {state: {action: probability}}) and value fonction (dictionary: {state: float})
    """
    
    # auxiliary function that computes the expected rewards for given state and actions
    def G_a(state, actions, V):
        R = game.rewards(state, actions)
        R_s_a = R.get(0)
        
        T_s_a = game.transition(state, actions)
        gamma = game.gamma()
        return R_s_a + gamma * sum(T_s_a[state2] * V[state2] for state2 in T_s_a.keys())
        
    # expected rewards matrix
    G = {state: \
             {actionB: \
                  {actionA : \
                       G_a(state, {0: actionA, 1: actionB}, V) \
                  for actionA in game.actions(0, state)} \
             for actionB in game.actions(1, state)} \
        for state in game.states()}
                    
    # for each state the associated static game is solved with linear programming to get the new V and pi
    new_V, pi = {}, {}
    for state in game.states():
        new_V[state], pi[state] = sgr.maximin(game, G[state], state)
        
    return pi, new_V


class Shapley_Agent(non_RL_Agent):
    """
    A Specific Agent class for Random (stationnary uniform policy) Agent
    """

    def __init__(self,game,playerID):
        super(Shapley_Agent,self).__init__(game,playerID)
        self.Q = {state: \
                     {action_B: \
                         {action_A : 0.0 for action_A in self.g.actions(self.playerID, state)} \
                     for action_B in self.g.actions((1-self.playerID), state)} \
                 for state in self.g.states()}
        self.V = {state: 1 for state in self.g.states()}
        self.epsilon = 10** -7

    def ecart(self,V1,V2):
        """
        return the maximal difference between the same-key values of two dictionaries and the sum of all differences
    
        :param V1, V2: dictionary {index: float}
        :rtype: maximal difference, sum of differences
        """
        m = max(abs(V1[state] - V2[state]) for state in V1.keys())
        s = sum(abs(V1[state] - V2[state]) for state in V1.keys())
        return m, s

    def compute_policy(self): #old shapley
        """
        Shapley's algorithm
        :rtype: player A strategy (dictionary: {state: {action: probability}}) and value fonction (dictionary: {state: float})
        """
      
        # V initialized
        V = {state: np.random.rand() for state in self.g.states()}
    
        # first iteration on value
        pi, V2 = new_V_and_pi(self.g,self.V)
        ite = 1
    
        # repeat until convergence
        m, _ = self.ecart(V, V2) 
        while m > self.epsilon:
            V = copy.deepcopy(V2)
            pi, V2 = new_V_and_pi(self.g, self.V)
            ite += 1
            m, _ = self.ecart(V, V2) 
        print("Number of iterations: "+str(ite))
        return pi, V2




