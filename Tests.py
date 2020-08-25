from Benchmarking import *


######################################################################
############################# VALIDATION #############################
######################################################################

# fonction permettant de calculer la distance entre 2 politiques : 
# somme des écarts des probas pour chaque état/action : (1/|S|)*sum_s (1/|A|)*(sum_a (pi1(s,a)-pi2(s,a))^2)
def distance(pi1,pi2):
    A = copy.deepcopy(pi1)
    B = { state : {action : (1/len(A[state].keys()))*(pi1[state][action]-pi2[state][action])**2 for action in A[state].keys() } for state in A.keys() }
    C = {state : sum(B[state].values()) for state in B.keys()}
    return sum(C.values())/len(C.keys())

# fonction permettant de calculer les différences (entre pi_t-1 et pi_t) d'une politique pi
def diff(pi): # pi = liste de politiques
    L = np.zeros(len(pi))
    for i in range(len(pi)-1):
        L[i] = distance(pi[i+1],pi[i])
    return L

def test_choice_parameters(): # test rationalité et convergence
    """
    Testing function for choosing parameters for RL Agents - two tests :
    * Rationaliy : Learning against optimal startegy GridWorld_Nash1_Player1_Agent : testing convergence towards GridWorld_Nash1_Player0_Agent
    * Convergence : Learning in self play : testing convergence towards stationnary policy
    """
    game = GridWorld() 
    playerA, playerB = game.players()[0], game.players()[1] #player ID

    Agent = WoLF_PHC_Agent(game, playerA)
    nb_iterations = 500000 #same for both tests
    timestamp = 1000

    ################################################
    ############ Test 1 : rationalité ##############
    ################################################
    Opponent_test1 = GridWorld_Nash1_Player1_Agent(game)
    policy_test1, pi_opponent1 = Scheduler(game,nb_iterations,timestamp,Agent,Opponent_test1)

    optimal_Agent = GridWorld_Nash1_Player0_Agent(game)
    optimal_Agent.compute_policy()
    pi_optimal = optimal_Agent.pi
    print(distance(policy_test1[-1],pi_optimal))

    Y = []
    for i in range(len(policy_test1)):
        Y.append(distance(policy_test1[i],pi_optimal))
    plt.plot(Y)
    plt.title("Test 1 : convergence contre optimal")
    plt.xlabel("nombre d'itérations (x1000)")
    plt.ylabel("erreur")
    plt.show() 

    ################################################
    ############ Test 2 : convergence ##############
    ################################################
    Agent2 = WoLF_PHC_Agent(game, playerA)
    Opponent_test2 = WoLF_PHC_Agent(game, playerB)
    policy_test2, pi_opponent2 = Scheduler(game,nb_iterations,timestamp,Agent2,Opponent_test2)
    Y2 = diff(policy_test2)
    plt.plot(Y2)
    plt.title("Test 2 : convergence vers stationnarité")
    plt.xlabel("nombre d'itérations (x1000)")
    plt.ylabel("diff")
    plt.show()

#test_choice_parameters()

def test(): # pour comparer 2 "types" d'Agents à la fois
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


######################################################################
###################### COMPARAISONS : GRAPHIQUES #####################
######################################################################

def convergence(game,nb_iterations,timestamp,A1,A2): # étant donné deux agents A1 et A2, fait l'apprentissage A1vsA2, et renvoit l'évolution de la convergence de A1
    pi1,pi2 = Scheduler(game,nb_iterations,timestamp,A1,A2)

    #distances between Nash 1 and 2
    optimal_Nash1_player0 = GridWorld_Nash1_Player0_Agent(game)
    optimal_Nash1_player0.compute_policy()
    optimal_Nash2_player0 = GridWorld_Nash2_Player0_Agent(game)
    optimal_Nash2_player0.compute_policy()
    d10 = []
    d20 = []
    for i in range(len(pi1)):
        d10.append(distance(pi1[i],optimal_Nash1_player0.pi))
        d20.append(distance(pi1[i],optimal_Nash2_player0.pi))
    return d10,d20

#Agents : 
#Y = Minimax_Q_Agent ; Q_Learning_Agent ; WoLF_PHC_Agent ; PD_WoLF_Agent ; EXORL_Agent ; NSCP_Agent
#Z = Random_Agent ; Shapley_Agent

#############################
########## Type 1 : #########
#############################
# Type 1 : X vs Z, with Z all other non_RL_Agents : 2 graphs, on for each non_RL_Agent, 6 types(colors) per graph

def Random_test():
    game = GridWorld()
    nb_iterations = 500000
    timestamp = 10000
    A0 = Random_Agent(game,1)

    # Minimax_Q_Agent = magenta
    A1 = Minimax_Q_Agent(game,0)
    A1_d10,A1_d20 = convergence(game,nb_iterations,timestamp,A1,A0)
    print("1 done")
    plt.plot(A1_d10, 'm', label = "Minimax-Q")
    plt.plot(A1_d20, 'm--')
    # Q_Learning_Agent = green
    A2 = Q_Learning_Agent(game,0)
    A2_d10,A2_d20 = convergence(game,nb_iterations,timestamp,A2,A0)
    print("2 done")
    plt.plot(A2_d10, 'g', label = "Q-Learning")
    plt.plot(A2_d20, 'g--')
    # WoLF_PHC_Agent = blue
    A3 = WoLF_PHC_Agent(game,0)
    A3_d10,A3_d20 = convergence(game,nb_iterations,timestamp,A3,A0)
    print("3 done")
    plt.plot(A3_d10, 'b', label = "WoLF-PHC")
    plt.plot(A3_d20, 'b--')
    # PD_WoLF_Agent = red
    A4 = PD_WoLF_Agent(game,0)
    A4_d10,A4_d20 = convergence(game,nb_iterations,timestamp,A4,A0)
    print("4 done")
    plt.plot(A4_d10, 'r', label = "PD-WoLF")
    plt.plot(A4_d20, 'r--')
    # EXORL_Agent = cyan
    A5 = EXORL_Agent(game,0)
    A5_d10,A5_d20 = convergence(game,nb_iterations,timestamp,A5,A0)
    print("5 done")
    plt.plot(A5_d10, 'c', label = "EXORL")
    plt.plot(A5_d20, 'c--')
    # NSCP_Agent = yellow
    A6 = NSCP_Agent(game,0)
    A6_d10,A6_d20 = convergence(game,nb_iterations,timestamp,A6,A0)
    print("6 done")
    plt.plot(A6_d10, 'y', label = "NSCP")
    plt.plot(A6_d20, 'y--')

    legend = plt.legend(loc='upper left', shadow=True)
    axes = plt.gca()
    axes.set_ylim([0.1,0.5])
    plt.show()

#Random_test()

def Shapley_test():
    game = GridWorld()
    nb_iterations = 500000
    timestamp = 10000
    A0 = Shapley_Agent(game,1)

    # Minimax_Q_Agent = magenta
    A1 = Minimax_Q_Agent(game,0)
    A1_d10,A1_d20 = convergence(game,nb_iterations,timestamp,A1,A0)
    print("1 done")
    plt.plot(A1_d10, 'm', label = "Minimax-Q")
    plt.plot(A1_d20, 'm--')
    # Q_Learning_Agent = green
    A2 = Q_Learning_Agent(game,0)
    A2_d10,A2_d20 = convergence(game,nb_iterations,timestamp,A2,A0)
    print("2 done")
    plt.plot(A2_d10, 'g', label = "Q-Learning")
    plt.plot(A2_d20, 'g--')
    # WoLF_PHC_Agent = blue
    A3 = WoLF_PHC_Agent(game,0)
    A3_d10,A3_d20 = convergence(game,nb_iterations,timestamp,A3,A0)
    print("3 done")
    plt.plot(A3_d10, 'b', label = "WoLF-PHC")
    plt.plot(A3_d20, 'b--')
    # PD_WoLF_Agent = red
    A4 = PD_WoLF_Agent(game,0)
    A4_d10,A4_d20 = convergence(game,nb_iterations,timestamp,A4,A0)
    print("4 done")
    plt.plot(A4_d10, 'r', label = "PD-WoLF")
    plt.plot(A4_d20, 'r--')
    # EXORL_Agent = cyan
    A5 = EXORL_Agent(game,0)
    A5_d10,A5_d20 = convergence(game,nb_iterations,timestamp,A5,A0)
    print("5 done")
    plt.plot(A5_d10, 'c', label = "EXORL")
    plt.plot(A5_d20, 'c--')
    # NSCP_Agent = yellow
    A6 = NSCP_Agent(game,0)
    A6_d10,A6_d20 = convergence(game,nb_iterations,timestamp,A6,A0)
    print("6 done")
    plt.plot(A6_d10, 'y', label = "NSCP")
    plt.plot(A6_d20, 'y--')

    legend = plt.legend(loc='upper left', shadow=True)
    axes = plt.gca()
    axes.set_ylim([0.1,0.5])
    plt.show()

#Shapley_test() # takes around 30minutes


#############################
########## Type 2 : #########
#############################

# Type 2 : X vs Y, with Y all other RL_Agents : 6 graphs, one for each RL Agent, 6 types(colors) per graph
#Minimax_Q_Agent ; Q_Learning_Agent ; WoLF_PHC_Agent ; PD_WoLF_Agent ; EXORL_Agent ; NSCP_Agent
#vs
#Minimax_Q_Agent ; Q_Learning_Agent ; WoLF_PHC_Agent ; PD_WoLF_Agent ; EXORL_Agent ; NSCP_Agent

def RL_vs_RL_test(A):
    game = GridWorld()
    nb_iterations = 500000
    timestamp = 10000

    # Minimax_Q_Agent = magenta
    A1 = Minimax_Q_Agent(game,1)
    A1_d10,A1_d20 = convergence(game,nb_iterations,timestamp,A,A1)
    print("1 done")
    plt.plot(A1_d10, 'm', label = "Minimax-Q")
    plt.plot(A1_d20, 'm--')
    # Q_Learning_Agent = green
    A2 = Q_Learning_Agent(game,1)
    A2_d10,A2_d20 = convergence(game,nb_iterations,timestamp,A,A2)
    print("2 done")
    plt.plot(A2_d10, 'g', label = "Q-Learning")
    plt.plot(A2_d20, 'g--')
    # WoLF_PHC_Agent = blue
    A3 = WoLF_PHC_Agent(game,1)
    A3_d10,A3_d20 = convergence(game,nb_iterations,timestamp,A,A3)
    print("3 done")
    plt.plot(A3_d10, 'b', label = "WoLF-PHC")
    plt.plot(A3_d20, 'b--')
    # PD_WoLF_Agent = red
    A4 = PD_WoLF_Agent(game,1)
    A4_d10,A4_d20 = convergence(game,nb_iterations,timestamp,A,A4)
    print("4 done")
    plt.plot(A4_d10, 'r', label = "PD-WoLF")
    plt.plot(A4_d20, 'r--')
    # EXORL_Agent = cyan
    A5 = EXORL_Agent(game,1)
    A5_d10,A5_d20 = convergence(game,nb_iterations,timestamp,A,A5)
    print("5 done")
    plt.plot(A5_d10, 'c', label = "EXORL")
    plt.plot(A5_d20, 'c--')
    # NSCP_Agent = yellow
    A6 = NSCP_Agent(game,1)
    A6_d10,A6_d20 = convergence(game,nb_iterations,timestamp,A,A6)
    print("6 done")
    plt.plot(A6_d10, 'y', label = "NSCP")
    plt.plot(A6_d20, 'y--')

    legend = plt.legend(loc='upper left', shadow=True)
    axes = plt.gca()
    axes.set_ylim([0.1,0.5])
    plt.show()

game=GridWorld()
#A1 = Minimax_Q_Agent(game,0) 
#RL_vs_RL_test(A1)
#A2 = Q_Learning_Agent(game,0) 
#RL_vs_RL_test(A2)
#A3 = WoLF_PHC_Agent(game,0) 
#RL_vs_RL_test(A3) 
#A4 = PD_WoLF_Agent(game,0) 
#RL_vs_RL_test(A4)
#A5 = EXORL_Agent(game,0) 
#RL_vs_RL_test(A5)
#A6 = NSCP_Agent(game,0) 
#RL_vs_RL_test(A6)


#############################
########## Type 3 : #########
#############################

# Type 3 : XvsX : 1 graph with 6 types (colors)

def Self_Play_test():
    game = GridWorld()
    nb_iterations = 500000
    timestamp = 10000

    # Minimax_Q_Agent = magenta
    A1 = Minimax_Q_Agent(game,0)
    O1 = Minimax_Q_Agent(game,1)
    A1_d10,A1_d20 = convergence(game,nb_iterations,timestamp,A1,O1)
    print("1 done")
    plt.plot(A1_d10, 'm', label = "Minimax-Q")
    plt.plot(A1_d20, 'm--')
    # Q_Learning_Agent = green
    A2 = Q_Learning_Agent(game,0)
    O2 = Q_Learning_Agent(game,1)
    A2_d10,A2_d20 = convergence(game,nb_iterations,timestamp,A2,O2)
    print("2 done")
    plt.plot(A2_d10, 'g', label = "Q-Learning")
    plt.plot(A2_d20, 'g--')
    # WoLF_PHC_Agent = blue
    A3 = WoLF_PHC_Agent(game,0)
    O3 = WoLF_PHC_Agent(game,1)
    A3_d10,A3_d20 = convergence(game,nb_iterations,timestamp,A3,O3)
    print("3 done")
    plt.plot(A3_d10, 'b', label = "WoLF-PHC")
    plt.plot(A3_d20, 'b--')
    # PD_WoLF_Agent = red
    A4 = PD_WoLF_Agent(game,0)
    O4 = PD_WoLF_Agent(game,1)
    A4_d10,A4_d20 = convergence(game,nb_iterations,timestamp,A4,O4)
    print("4 done")
    plt.plot(A4_d10, 'r', label = "PD-WoLF")
    plt.plot(A4_d20, 'r--')
    # EXORL_Agent = cyan
    A5 = EXORL_Agent(game,0)
    O5 = EXORL_Agent(game,1)
    A5_d10,A5_d20 = convergence(game,nb_iterations,timestamp,A5,O5)
    print("5 done")
    plt.plot(A5_d10, 'c', label = "EXORL")
    plt.plot(A5_d20, 'c--')
    # NSCP_Agent = yellow
    A6 = NSCP_Agent(game,0)
    O6 = NSCP_Agent(game,1)
    A6_d10,A6_d20 = convergence(game,nb_iterations,timestamp,A6,O6)
    print("6 done")
    plt.plot(A6_d10, 'y', label = "NSCP")
    plt.plot(A6_d20, 'y--')

    legend = plt.legend(loc='upper left', shadow=True)
    axes = plt.gca()
    axes.set_ylim([0.1,0.5])
    plt.show()

#Self_Play_test()


######################################################################
########################### TESTS : DUEL #############################
######################################################################

def Duel():
    game = GridWorld()
    nb_iterations = 100#500000
    timestamp = (nb_iterations-1)
    A1 = WoLF_PHC_Agent(game,0)
    A2 = PD_WoLF_Agent(game,1)

    WoLF_PHC_policies = []
    PD_WoLF_policies = []

    # Learning :

    # Random_Agent
    B1 = Random_Agent(game,1)
    A1_random, random1 = Scheduler(game,nb_iterations,timestamp,A1,B1)
    WoLF_PHC_policies.append(A1_random[-1])
    print("WoLF-PHC vs Random Learning done")
    b1 = Random_Agent(game,0)
    random2, A2_random = Scheduler(game,nb_iterations,timestamp,b1,A2)
    PD_WoLF_policies.append(A2_random[-1])
    print("PD-WoLF vs Random Learning done")

    # Shapley_Agent
    B2 = Shapley_Agent(game,1)
    A1_shapley, shapley1 = Scheduler(game,nb_iterations,timestamp,A1,B2)
    WoLF_PHC_policies.append(A1_shapley[-1])
    print("WoLF-PHC vs Shapley Learning done")
    b2 = Shapley_Agent(game,0)
    shapley2, A2_shapley = Scheduler(game,nb_iterations,timestamp,b2,A2)
    PD_WoLF_policies.append(A2_shapley[-1])
    print("PD-WoLF vs Shapley Learning done")

    # Q_Learning_Agent
    B3 = Q_Learning_Agent(game,1)
    A1_q_learning, q_learning1 = Scheduler(game,nb_iterations,timestamp,A1,B3)
    WoLF_PHC_policies.append(A1_q_learning[-1])
    print("WoLF-PHC vs Q-Learning Learning done")
    b3 = Q_Learning_Agent(game,0)
    q_learning2, A2_q_learning = Scheduler(game,nb_iterations,timestamp,b3,A2)
    PD_WoLF_policies.append(A2_q_learning[-1])
    print("PD-WoLF vs Q-Learning Learning done")

    # Minimax_Q_Agent
    B4 = Minimax_Q_Agent(game,1)
    A1_minimax, minimax1 = Scheduler(game,nb_iterations,timestamp,A1,B4)
    WoLF_PHC_policies.append(A1_minimax[-1])
    print("WoLF-PHC vs Minimax-Q Learning done")
    b4 = Minimax_Q_Agent(game,0)
    minimax2, A2_minimax = Scheduler(game,nb_iterations,timestamp,b4,A2)
    PD_WoLF_policies.append(A2_minimax[-1])
    print("PD-WoLF vs Minimax-Q Learning done")

    # EXORL_Agent
    B5 = EXORL_Agent(game,1)
    A1_exorl, exorl1 = Scheduler(game,nb_iterations,timestamp,A1,B5)
    WoLF_PHC_policies.append(A1_exorl[-1])
    print("WoLF-PHC vs EXORL Learning done")
    b5 = EXORL_Agent(game,0)
    exorl2, A2_exorl = Scheduler(game,nb_iterations,timestamp,b5,A2)
    PD_WoLF_policies.append(A2_exorl[-1])
    print("PD-WoLF vs EXORL Learning done")

    # NSCP_Agent
    B6 = NSCP_Agent(game,1)
    A1_nscp, nscp1 = Scheduler(game,nb_iterations,timestamp,A1,B6)
    WoLF_PHC_policies.append(A1_nscp[-1])
    print("WoLF-PHC vs NSCP Learning done")
    b6 = NSCP_Agent(game,0)
    nscp2, A2_nscp = Scheduler(game,nb_iterations,timestamp,b6,A2)
    PD_WoLF_policies.append(A2_nscp[-1])
    print("PD-WoLF vs NSCP Learning done")

    # PD_WoLF_Agent
    B7 = PD_WoLF_Agent(game,1)
    A1_pd_wolf, pd_wolf1 = Scheduler(game,nb_iterations,timestamp,A1,B7)
    WoLF_PHC_policies.append(A1_pd_wolf[-1])
    print("WoLF-PHC vs PD_WoLF Learning done")
    b7 = PD_WoLF_Agent(game,0)
    pd_wolf2, A2_pd_wolf = Scheduler(game,nb_iterations,timestamp,b7,A2)
    PD_WoLF_policies.append(A2_pd_wolf[-1])
    print("PD-WoLF vs self Learning done")

    # WoLF_PHC_Agent
    B8 = WoLF_PHC_Agent(game,1)
    A1_wolf_phc, wolf_phc1 = Scheduler(game,nb_iterations,timestamp,A1,B8)
    WoLF_PHC_policies.append(A1_wolf_phc[-1])
    print("WoLF-PHC vs self Learning done")
    b8 = WoLF_PHC_Agent(game,0)
    wolf_phc2, A2_wolf_phc = Scheduler(game,nb_iterations,timestamp,b8,A2)
    PD_WoLF_policies.append(A2_wolf_phc[-1])
    print("PD-WoLF vs WoLF-PHC Learning done")

    # Battle :
    # moyenne sur 3x (battle with 1000 rounds) du pourcentage de parties gagnées

    score1 = np.zeros((8,8)) #wolf_phc scores
    score2 = np.zeros((8,8)) #pd_wolf scores
    rapport = np.zeros((8,8))
    i,j = 0,0

    print("Battle")
    nbplay = 1000

    for policy1 in WoLF_PHC_policies:
        for policy2 in PD_WoLF_policies:
            for k in range(3):
                wins1,wins2 = affrontement(game,policy1,policy2,nbplay)
                score1[i,j] += (wins1/(wins1+wins2+1))*100
                score2[i,j] += (wins2/(wins1+wins2+1))*100
            score1[i,j] = np.round(score1[i,j]/3,1)
            score2[i,j] = np.round(score2[i,j]/3,1)
            if j < 7 :
                j += 1
            else:
                j=0
        i += 1
    
    return(score1,score2)

score1,score2 = Duel()
print(score1)
print(score2)
