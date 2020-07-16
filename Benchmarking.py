# -*- coding: utf-8 -*-

from RockPaperScissors import RockPaperScissors
from Soccer import Soccer
from GridWorld import GridWorld
from Battle import Battle
import time
import random
import numpy as np
from Agents import Agent
from WoLF_PHC_Agent import WoLF_PHC_Agent
from Minimax_Q_Agent import Minimax_Q_Agent
from Random_Agent import Random_Agent

def benchmarking(game, playerA, playerB, player1, opponent1, player2, opponent2, nb_iterations, display_nb):
    """
    A training function
    
    :param playerA: maximizing player ID
    :param playerB: minimizing player ID
    :param player1: Agent
    :param opponent1: Training opponent Agent for player1
    :param player2: Agent
    :param opponent2: Training opponent Agent for player2
    :param nb_iteration: number of iterations for training
    :param display_nb: integer (number of iterationsto print at every 'display_nb' iterations)
    
    :rtype: player1 strategy (dictionary: {state: {action: probability}}) and player2 strategy (dictionary: {state: {action: probability}})
    """
    # train :
    policy1_list,V_1_list = player1.training(game,playerA,playerB,nb_iterations,opponent1,display_nb)
    policy2_list,V_2_list = player2.training(game,playerA,playerB,nb_iterations,opponent2,display_nb)
    # analyse : time, convergence to stationnary (pi : ecarts successifs tendent vers 0), convergence to nash (regarder V)
    #todo
    return policy1_list[-1], policy2_list[-1] #final policies

def affrontement(game,playerA,playerB,policy1,policy2,nbplay):
    """
    Simulation of game
    
    :param playerA: maximizing player ID
    :param playerB: minimizing player ID
    :param policy1: player1 strategy (dictionary: {state: {action: probability}}) 
    :param policy2: player2 strategy (dictionary: {state: {action: probability}}) 
    :param nb_play: integer number of iterations
    """
    #uses battle to have policy A and B play against each other
    policy  = {playerA : policy1 , playerB : policy2 }
    battle = Battle(game,nbplay,policy )
    battle.simulation()
    print("total_rewards")
    print(battle.total_rewards)
    wins1,wins2 = battle.get_winners()
    print("player1 won :",wins1,"times out of ",nbplay,"games")
    print("player2 won :",wins2,"times out of ",nbplay,"games")


def test():
    """
    Testing function : Training for a given game and 2 agents, and Battle between both agents
    """

    #Choose Game :
    #game = Soccer()
    game = RockPaperScissors()
    #game = GridWorld()

    playerA, playerB = game.players()[0], game.players()[1] #player ID

    #Choose Player A and training opponent (agents):
    player1 = WoLF_PHC_Agent(game, playerA, playerB)
    #opponent1 = WoLF_PHC_Agent(game, playerB, playerA)
    opponent1 = Random_Agent(game, playerB, playerA)

    #Choose Player B and training opponent (agents):
    #player2 = WoLF_PHC_Agent(game, playerA, playerB)
    #opponent2 = WoLF_PHC_Agent(game, playerB, playerA)
    player2 = Minimax_Q_Agent(game, playerA, playerB)
    opponent2 = Minimax_Q_Agent(game, playerB, playerA)

    #Train policies :
    nb_iterations = 10000
    display_nb = 1000
    policy1, policy2 = benchmarking(game, playerA, playerB, player1, opponent1, player2, opponent2, nb_iterations, display_nb)

    #Battle :
    print("Battle")
    nbplay = 10000
    affrontement(game,playerA,playerB,policy1,policy2,nbplay) 
    #score = affrontement(policy1,policy2) 
    #print(score)

test()
