# -*- coding: utf-8 -*-

# Test Game RPS

from RockPaperScissors import RockPaperScissors
from Soccer import Soccer
from GridWorld import GridWorld
from Shapley import new_V_and_pi
from Shapley import ecart
from Shapley import shapley
from Littman import MinimaxQ
from WoLF_PHC import WoLF_PHC
from Battle import Battle
import time
import random
import numpy as np

def train(game,agent,adversary_policy): #renvoit la politique (et V) pour un type d'agent donné
    p0, p1 = game.players()[0], game.players()[1]

    if (agent == "Random") :
        pi = {state: {action: 1/len(game.actions(p1, state)) for action in game.actions(p1, state)} for state in game.states()}
        return pi

    if (agent == "Shapley") :
        #shapleyepsilon = 0.0
        shapleyepsilon = 10** -7 
        start_time = time.time()
        pi,V = shapley(game, epsilon = shapleyepsilon,playerA = p1,playerB = p0)
        print("Time shapley : ",time.time() - start_time)
        return pi,V

    if (agent == "Littman") :
        explor = .3
        decay = .01 ** (1. / ( 2 *10**4))
        display = 10000
        iteration = 2* 10**4
        start_time = time.time()
        pi,V = MinimaxQ(game,explor,decay,display,iteration,playerA=p0,playerB=p1,policyplayerB=adversary_policy)
        print("Time Littman : ",time.time() - start_time)
        return pi,V

    if (agent == "WolF_PHC") :
        explor = .3
        decay = .01 ** (1. / ( 2 *10**4))
        delta_win = 0.0025
        delta_lose = 0.01
        display = 100
        iteration = 1000
        start_time = time.time()
        pi_wolf,V_wolf = WoLF_PHC(game,explor,decay,delta_win,delta_lose,display,iteration,playerA=p0,playerB=p1,policyplayerB=adversary_policy)
        print("Time WolF : ",time.time() - start_time)
        return pi_wolf,V_wolf

    if (agent == "Qlearning") :
        pi = 0
        return pi


def Benchmarking(Game,nbplay,agentA,agentB):
    if (Game == "RockPaperScissors"):
        game = RockPaperScissors()
    if (Game == "Soccer") :
        game = Soccer()
    if (Game == "GridWorld") :
        game = GridWorld()
    p0, p1 = game.players()[0], game.players()[1]
    RandomPlayer =  {state: {action: 1/len(game.actions(p1, state)) for action in game.actions(p1, state)} for state in game.states()}
    adversary_policy = RandomPlayer
    #policyA,V_A = train(game,agentA,adversary_policy)
    #policyB = train(game,agentB,adversary_policy)
    policyA = RandomPlayer
    policyB = RandomPlayer
    policy  = {p0 : policyA , p1 :policyB }

    battle = Battle(game,nbplay,policy )
    battle.simulation()
    print("total_rewards")
    print(battle.total_rewards)
    print("winers :",battle.get_winners()," nombre fois arrive en etat but",battle.but )
    #+ faire les autres analyses/graphiques ici :
    #new_pi , new_V= new_V_and_pi(game, V_shap,playerA = p1,playerB = p0)
    #print("ecart entre new_V et val de Shapley : ")
    #print(ecart(new_V,V_shap)) 

#pour lancer on utilise :
agent = ["Random","Shapley", "Littman", "WolF_PHC", "Qlearning"]
game = ["RockPaperScissors","Soccer","GridWorld"]

Benchmarking(game[2] , 100 , agent[0] , agent[0])
