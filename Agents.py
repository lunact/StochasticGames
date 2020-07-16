
import numpy as np
import random

def roulette(dict):
    """
    randomly return one of the dictionary's key according to its probability (its value)
    
    :param dict: dictionary {index: probability}
    :rtype: index
    """
    rnd = np.random.rand()
    tmp = 0
    for s, p in dict.items():
        tmp += p
        if tmp >= rnd:
            return s

class Agent:
    """
    Generic class for agents
    """

    def update(self,playerA,playerB,playerB_policy):
        """Returns updated policy and V table
        
        :rtype: strategy (dictionary: {state: {action: probability}}) and value fonction (dictionary: {state: float})
        """
        raise(NotImplementedError)


    def play(self,s,playerA,playerB,playerB_policy):
        """ Choice of actions for players, and obtain rewards
        
        :rtype: action player A and player B (integers/strings) , joint actions (dictionary: {playerA: a, playerB: o}) and rewards (dictionary: (key: player; value: reward))
        """
        raise(NotImplementedError)


    def training(self,game,playerA,playerB,nb_iterations,opponent,display_nb):
        """Returns trained policy and V table
        
        :rtype: list of strategies (dictionary: {state: {action: probability}}) and list of value fonction (dictionary: {state: float})
        """
        s = roulette(game.initial_state())
        policy_opponent = opponent.pi
        k = 0
        policy_list = []
        V_list = []
        while k < nb_iterations :
            a,o,actions,rew = self.play(s,playerA,playerB,policy_opponent)
            s2 = roulette(game.transition(s, actions))
            policy_player,V_player = self.update(s,s2,k,a,o,rew,playerA,playerB,policy_opponent)
            policy_opponent,V_opponent = opponent.update(s,s2,k,o,a,rew,playerB,playerA,policy_player)
            s = s2
            if(k % display_nb == 0):
                print("iteration: ", k)
                policy_list.append(policy_player)
                V_list.append(V_player)
            k += 1
        return policy_list,V_list



