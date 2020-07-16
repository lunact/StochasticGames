from Agents import Agent
import StaticGameResolution as sgr
from WoLF_PHC_Agent import WoLF_PHC_Agent
from Minimax_Q_Agent import Minimax_Q_Agent
from Soccer import Soccer
import numpy as np
import random

class Random_Agent(Agent):
    """
    A Specific Agent class for Random (stationnary uniform policy) Agent
    """
    
    def __init__(self,game,playerA,playerB):
        self.g = game
        self.pi = {state: {action: 1/len(game.actions(playerA, state)) for action in game.actions(playerA, state)} for state in game.states()}

    def update(self,s,s2,k,a,o,rew,playerA,playerB,playerB_policy):
        V = None
        return self.pi,V

#game = Soccer()
#playerA, playerB = game.players()[0], game.players()[1]
#test_agent =  Minimax_Q_Agent(game, playerA,playerB)
#opponent = Random_Agent(game, playerB,playerA)
#pi,V = test_agent.training(game,playerA,playerB,100,opponent,10)