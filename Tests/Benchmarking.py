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
    """Returns trained policies
    
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
    return policyA, policyB



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




def test():
    """
    Testing function : Training for a given game and 2 agents, and Battle between both agents
    """

    #############################################
    ################ Choose Game ################
    #############################################
    #game = Soccer()
    #game = RockPaperScissors()
    game = GridWorld()

    playerA, playerB = game.players()[0], game.players()[1] #player ID

    #############################################
    ### Choose Player 1 and training opponent ###
    #############################################
    player1 = WoLF_PHC_Agent(game, playerA)
    opponent1 = WoLF_PHC_Agent(game, playerB)

    #############################################
    ### Choose Player 2 and training opponent ###
    #############################################
    player2 = Minimax_Q_Agent(game, playerB)
    opponent2 = Random_Agent(game, playerA)


    #############################################
    ############## Train Policies ###############
    #############################################
    nb_iterations = 500000
    timestamp = 1000

    start_time = time.time()
    policy1, policyb = Scheduler(game,nb_iterations,timestamp,player1,opponent1)
    print("Learning Time Player 1: ",time.time() - start_time)

    start_time = time.time()
    policyc, policy2 = Scheduler(game,nb_iterations,timestamp,opponent2,player2)
    print("Learning Time Player 2: ",time.time() - start_time)


    #policy 1 : distances between Nash 1 and 2
    optimal_Nash1_player0 = GridWorld_Nash1_Player0_Agent(game)
    optimal_Nash1_player0.compute_policy()
    optimal_Nash2_player0 = GridWorld_Nash2_Player0_Agent(game)
    optimal_Nash2_player0.compute_policy()
    d10 = []
    d20 = []
    for i in range(len(policy1)):
        d10.append(distance(policy1[i],optimal_Nash1_player0.pi))
        d20.append(distance(policy1[i],optimal_Nash2_player0.pi))

    #policy 2 : distances between Nash 1 and 2
    optimal_Nash1_player1 = GridWorld_Nash1_Player1_Agent(game)
    optimal_Nash1_player1.compute_policy()
    optimal_Nash2_player1 = GridWorld_Nash2_Player1_Agent(game)
    optimal_Nash2_player1.compute_policy()
    d11 = []
    d21 = []
    for i in range(len(policy2)):
        d11.append(distance(policy2[i],optimal_Nash1_player1.pi))
        d21.append(distance(policy2[i],optimal_Nash2_player1.pi))

    #plot :
    plt.plot(d10, 'b')
    plt.plot(d20, 'b--')
    plt.plot(d11, 'r')
    plt.plot(d21, 'r--')
    plt.show()

    #Battle :
    print("Battle")
    nbplay = 1000
    affrontement(game,policy1[-1],policy2[-1],nbplay)
    return(policy1,policy2)

#test()