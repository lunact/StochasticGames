# -*- coding: utf-8 -*-

import numpy as np
import random

#on commence avec 2 joueurs : playerA, playerB, on pourra essayer de généraliser plus tard
# delta_win = 0.05
# delta_lose = 0.2
# alpha = 0.9 (for littman it's 1.)
# explor = epsilon (for epsilon-greedy exploration)
# decay=proportionnel à 1/k : is the the way alhpha decays (but also for delta_win and delta_lose because must be proportional)

def roulette(dict):
#def roulette(self, dict):
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

def WoLF_PHC(game, explor, decay, delta_win, delta_lose, display, iteration, playerA, playerB, policyplayerB):
    """
    WoLF-PHC (Win or Learn Fast - Policy Hill Climbing)
    
    :param game: NullSum2PlayerStochasticGame
    :param explor: float
    :param decay: float
    :param display: integer (the number of the current iteration will be printed at every 'display' iterations)
    :param iteration: number of iterations
    :param playerA: maximizing player ID
    :param playerB: minimizing player ID
    :param policyplayerB: player B strategy (dictionary: {state: {action: probability}})
    :rtype: player A strategy (dictionary: {state: {action: probability}}) and value fonction (dictionary: {state: float})
    """

    # 0) initialisation
    Q = {state: \
             {action_B: \
                  {action_A : 0.0 for action_A in game.actions(playerA, state)}\
             for action_B in game.actions(playerB, state)} \
        for state in game.states()}
    
    pi = {state: {action: 1/len(game.actions(playerA, state)) for action in game.actions(playerA, state)} for state in game.states()}
    pi_bar = {state: {action: 1/len(game.actions(playerA, state)) for action in game.actions(playerA, state)} for state in game.states()}
    alpha = 1.
    print(game.initial_state())
    s = roulette(game.initial_state())
    k = 0
    delta = delta_lose

    #boucle
    while(k <= iteration): 
        # 1) voir si on fait greedy ou pas
        a = random.choice(game.actions(playerA, s)) if np.random.rand() < explor else roulette(pi[s])
        # 2) on fait l'action a choisi par 1), on observe la récompense et le nouvel état x
        o = roulette(policyplayerB[s]) #action joueurB
        actions = {playerA: a, playerB: o}
        # on récupère la récompense
        R = game.rewards(s, actions)
        rew = R.get(playerA)
        # on observe le nouvel état
        s2 = roulette(game.transition(s, actions))

        # 3) on met a jour Q
        Q[s][o][a] = (1-alpha)*Q[s][o][a] + alpha*(rew + game.gamma()*max(list(Q[s][o].values())))

        k += 1
        if(k % display == 0):
            print("iteration: ", k)

        # 4) on calcule la politique "moyenne"
        for action in game.actions(playerA, s):
            pi_bar[s][action] = (1/k)*(pi[s][action] - pi_bar[s][action])
        # 5) on détermine si on gagne/perd et on choisi delta
        #pi_bar.items
        #if sum(pi[s].values()*Q[s][o].values()) > sum(pi_bar[s].values()*Q[s][o].values()) :
        if sum([p*q for p,q in zip(pi[s].values(),Q[s][o].values())]) > sum([p*q for p,q in zip(pi_bar[s].values(),Q[s][o].values())]) :
            delta = delta_win
        else :
            delta = delta_lose
        # 6) on met à jour la politique pi (pi[s][a]=...)
        argmax = game.actions(playerA, s)[np.argmax(list(Q[s][o].values()))]
        for action in game.actions(playerA, s):
            if action == argmax:
                pi[s][action] = pi[s][action] + delta
                pi[s][action] = min(pi[s][action],1.) #pour restreindre à une proba
            else :
                pi[s][action] = pi[s][action] - delta/(len(game.actions(playerA, s))-1)
                pi[s][action] = max(0.,pi[s][action]) #pour restreindre à une proba
    
        # 7) decay et mises à jour
        s = s2
        alpha *= decay
        delta_win *= decay #(?)
        delta_lose *= decay #(?)
        
    print("Learning rate at the end of the run: ", alpha)
    return pi #,V (rajouter V plus tard quand le code marche :p)
    
