# -*- coding: utf-8 -*-
from Random_Agent import Random_Agent
from Shapley_Agent import Shapley_Agent
from GridWorld_Nash1_Player0_Agent import GridWorld_Nash1_Player0_Agent
from GridWorld_Nash1_Player1_Agent import GridWorld_Nash1_Player1_Agent
from GridWorld_Nash2_Player0_Agent import GridWorld_Nash2_Player0_Agent
from GridWorld_Nash2_Player1_Agent import GridWorld_Nash2_Player1_Agent
from Minimax_Q_Agent import Minimax_Q_Agent
from Q_Learning_Agent import Q_Learning_Agent
from WoLF_PHC_Agent import WoLF_PHC_Agent
from PD_WoLF_Agent import PD_WoLF_Agent
from EXORL_Agent import EXORL_Agent
from NSCP_Agent import NSCP_Agent

from RockPaperScissors import RockPaperScissors
from Soccer import Soccer
from GridWorld import GridWorld
from Battle import Battle
import time
import random
import numpy as np
import copy
import matplotlib.pyplot as plt

def Scheduler(game,nb_iterations,timestamp,agentA,agentB):
    """Returns trained policy and V table
    
    :rtype: list of strategies (dictionary: {state: {action: probability}})
    """
    agentA.compute_policy()
    agentB.compute_policy()
    s = agentA.roulette(game.initial_state())
    policyA = []
    policyB = []
    policyA.append(copy.deepcopy(agentA.pi))
    policyB.append(copy.deepcopy(agentB.pi))
    policyA[0][((0,0),(0,2))]

    for k in range(nb_iterations):
        if (k % timestamp == 0 ):
            policyA.append(copy.deepcopy(agentA.pi))
            policyB.append(copy.deepcopy(agentB.pi))

        a = agentA.play(s,0)
        o = agentB.play(s,1)
        actions = {0: a, 1: o}
        s2 = agentA.roulette(game.transition(s, actions))
        R = game.rewards(s, actions)
        rewA = R.get(0)
        rewB = R.get(1)
        agentA.learn(s,s2,k,a,o,rewA,rewB,agentB.pi)
        agentB.learn(s,s2,k,o,a,rewB,rewA,agentA.pi)
        s = s2
    return policyA, policyB #agentA.pi, agentB.pi

def affrontement(game,policy1,policy2,nbplay):
    """
    Simulation of game
    
    :param policy1: player1 strategy (dictionary: {state: {action: probability}}) 
    :param policy2: player2 strategy (dictionary: {state: {action: probability}}) 
    :param nb_play: integer number of iterations
    """
    #uses battle to have policy 1 and 1 play against each other
    policy  = {0 : policy1 , 1 : policy2 }
    battle = Battle(game,nbplay,policy)
    battle.simulation()
    #print("total_rewards")
    #print(battle.total_rewards)
    wins1,wins2 = battle.get_winners()
    #print("player1 won :",wins1,"times out of",nbplay,"games")
    #print("player2 won :",wins2,"times out of",nbplay,"games")
    return(wins1,wins2)