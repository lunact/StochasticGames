from non_RL_Agent import non_RL_Agent
import numpy as np
import random
import copy
from gurobipy import *

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

    def maximin(self, state):
        """
        solve the static game associated with one state of a null sum 2-player stochastic game
    
        :param game: NullSum2PlayerStochasticGame
        :param expected_rewards: dictionary {player B action : {player A action : reward}}
        :param state: state ID
        :rtype: state value (float) and player A strategy (dictionary: {action: probability})
        """
    
        try:
            # Model created
            m = Model("jeu_simple")
            m.setParam('OutputFlag', False) # no console print
            # Variables
            V = m.addVar(lb = -GRB.INFINITY, vtype=GRB.CONTINUOUS, name="V") # V can be negative
            pi_s = np.array([m.addVar(vtype=GRB.CONTINUOUS, name="p_"+str(a)) for a in self.g.actions(self.playerID, state)])
            m.update()
            # Objective
            m.setObjective(V, GRB.MAXIMIZE)
            # Constraints
            for action1 in self.g.actions(1, state):
                # conversion dictionary > np.array while preserving the same order for actions
                expected_rewards_action1 = np.array([self.Q[state][action1][a] for a in self.g.actions(self.playerID, state)])
                m.addConstr(V <= np.dot(pi_s, expected_rewards_action1), "")
            m.addConstr(sum(pi_s) >= 1.0, "")
            m.addConstr(sum(pi_s) <= 1.0, "")
            m.addConstr(sum(pi_s), GRB.EQUAL, 1.0, "")
            for pi_s_a in pi_s:
                m.addConstr(pi_s_a >= 0, "")
            # Solving
            m.optimize()
            return(V.x, {action0: pi_s[i].x for i, action0 in enumerate(self.g.actions(0, state))})
    
        except GurobiError:
            print('Error reported')

    def new_V_and_pi(self):
        """
        compute the next value of V based on its current value
    
        :param game: NullSum2PlayerStochasticGame
        :param V: current value fonction (dictionary: {state: float})
        :rtype: player 0 strategy (dictionary: {state: {action: probability}}) and value fonction (dictionary: {state: float})
        """
    
        # auxiliary function that computes the expected rewards for given state and actions
        def G_a(state, actions, V):
            R = self.g.rewards(state, actions)
            R_s_a = R.get(0)
        
            T_s_a = self.g.transition(state, actions)
            gamma = self.g.gamma()
            return R_s_a + gamma * sum(T_s_a[state2] * V[state2] for state2 in T_s_a.keys())
        
        # expected rewards matrix
        G = {state: \
                 {actionB: \
                      {actionA : \
                           G_a(state, {0: actionA, 1: actionB}, self.V) \
                      for actionA in self.g.actions(0, state)} \
                 for actionB in self.g.actions(1, state)} \
            for state in self.g.states()}
                    
        # for each state the associated static game is solved with linear programming to get the new V and pi
        new_V, pi = {}, {}
        for state in self.g.states():
            new_V[state], pi[state] = self.maximin(state)
        
        return pi, new_V

    def compute_policy(self): #old shapley
        """
        Shapley's algorithm
        :rtype: player A strategy (dictionary: {state: {action: probability}}) and value fonction (dictionary: {state: float})
        """
      
        # V initialized
        V = {state: np.random.rand() for state in self.g.states()}
    
        # first iteration on value
        pi, V2 = self.new_V_and_pi()
        ite = 1
    
        # repeat until convergence
        m, _ = self.ecart(V, V2) 
        while m > self.epsilon:
            V = copy.deepcopy(V2)
            pi, V2 = self.new_V_and_pi()
            ite += 1
            m, _ = self.ecart(V, V2) 
        print("Number of iterations: "+str(ite))
        return pi, V2




