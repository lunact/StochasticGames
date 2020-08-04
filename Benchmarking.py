# -*- coding: utf-8 -*-
from Agents import Agent
from RL_Agent import RL_Agent
from non_RL_Agent import non_RL_Agent
from WoLF_PHC_Agent import WoLF_PHC_Agent
from Minimax_Q_Agent import Minimax_Q_Agent
from Random_Agent import Random_Agent
from Shapley_Agent import Shapley_Agent

from RockPaperScissors import RockPaperScissors
from Soccer import Soccer
from GridWorld import GridWorld
from Battle import Battle
import time
import random
import numpy as np

def Scheduler(game,nb_iterations,agentA,agentB):
    """Returns trained policy and V table
    
    :rtype: list of strategies (dictionary: {state: {action: probability}}) and list of value fonction (dictionary: {state: float})
    """
    agentA.compute_policy()
    agentB.compute_policy()
    s = agentA.roulette(game.initial_state())
    for k in range(nb_iterations):
        a = agentA.play(s,0)
        o = agentB.play(s,1)
        actions = {0: a, 1: o}
        s2 = agentA.roulette(game.transition(s, actions))
        R = game.rewards(s, actions)
        rewA = R.get(0)
        rewB = R.get(1)
        agentA.learn(s,s2,k,a,o,rewA,agentB.pi)
        agentB.learn(s,s2,k,o,a,rewB,agentA.pi)
        s = s2
    return agentA.pi, agentB.pi #les V aussi

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
    #game = RockPaperScissors()
    game = GridWorld()

    playerA, playerB = game.players()[0], game.players()[1] #player ID

    #Choose Player 1 and training opponent :
    #player1 = WoLF_PHC_Agent(game, 0)
    #opponent1 = WoLF_PHC_Agent(game, playerB, playerA)
    #opponent1 = Random_Agent(game, 1)

    #Choose Player 2 and training opponent :
    #player2 = WoLF_PHC_Agent(game, playerA, playerB)
    #opponent2 = WoLF_PHC_Agent(game, playerB, playerA)
    #player2 = Minimax_Q_Agent(game, 0)
    #opponent2 = Minimax_Q_Agent(game, 1)

    #Train policies :
    #nb_iterations = 10000
    #display_nb = 1000
    #policy1, policyb = Scheduler(game,nb_iterations,player1,opponent1)
    #policy2, policyc = Scheduler(game,nb_iterations,player2,opponent2)




    # OR for simultaneous learning, choose player1 and player2 directly :
    player1 = WoLF_PHC_Agent(game, playerA)
    player2 = Random_Agent(game, playerB)
    #Train policies :
    nb_iterations = 1000
    policy1, policy2 = Scheduler(game,nb_iterations,player1,player2)

    #Battle :
    print("Battle")
    nbplay = 1000
    affrontement(game,policy1,policy2,nbplay)
    #score = affrontement(policy1,policy2) 
    #print(score)

test()
