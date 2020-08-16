
from StochasticGame import StochasticGame

class GridWorld(StochasticGame):
    """
    Example of a null sum 2-player stochastic game: simplified football game (cf. Littman 1994)
    """
    
    movement = {"Up":(1, 0), "Down":(-1, 0), "Left":(0, -1), "Right":(0, 1)}
    list_actions = list(movement.keys())
    
    def players(self):
        return [0, 1]
    
    def __init__(self):
        self.cells = [(i, j) for i in range(3) for j in range(3)]    
        self.list_states = [(c1, c2) for c1 in self.cells for c2 in self.cells if (c1 != c2 and c1 != (2,1) and c2 != (2,1))]
        self.starting_positions = ((0, 0), (0, 2))
        self.goal_position = (2,1)
        

    def next_position(self, position, action):
        #return (position[0]+self.movement[action][0],position[1]+self.movement[action][1])
        return tuple(map(sum, zip(position, self.movement[action])))
    
    
    def states(self):
        return self.list_states
    
    
    def actions(self, player, state): # renvoit actions possibles selon le joueur et l'état courant
        def allowed(action):
            position = self.next_position(state[player], action)
            return position in self.cells or (position == self.goal_position)
        return [action for action in self.list_actions if allowed(action)]
    
    
    def gamma(self):
        return .9
    
    
    def initial_state(self):
        return {self.starting_positions : 1 }

    def indicatrice(self,pos,dest):
        if (pos == (0,0) and dest == (1,0)) or (pos == (0,2) and dest == (1,2)):
            return 1
        else:
            return 0
    
    def transition(self, state, actions): #renvoit dictionnaire {état:proba}
        # positions actuelles :
        pos1, pos2 = state
        # positions potentielles :
        dest1 = self.next_position(pos1, actions[0])
        dest2 = self.next_position(pos2, actions[1])

        # les différentes situations possibles :
        # goal :
        if (dest1 == self.goal_position) or (dest2 == self.goal_position):
            return self.initial_state()
        # si même destination :
        if (dest1 == dest2):
            return { (pos1,pos2):1 }
        # tout le reste
        else :
            if (dest1 == pos2) :
                if ((dest1 == (0,2) and pos2 == (0,2)) or (dest1 == (0,0) and pos2 == (0,0))) :
                    A = 0
                else :
                    A = (1-0.5*self.indicatrice(pos1,dest1))*(0.5*self.indicatrice(pos2,dest2))
            else :
                A = 0
            if (pos1 == dest2) :
                if ((pos1==(0,0) and dest2==(0,0)) or (pos1==(0,2) and dest2==(0,2))) :
                    B = 0
                else :
                    B = (0.5*self.indicatrice(pos1,dest1))*(1-0.5*self.indicatrice(pos2,dest2))
            else :
                B = 0
            C = (1-0.5*self.indicatrice(pos1,dest1))*(1-0.5*self.indicatrice(pos2,dest2))

            sol = { (pos1,pos2):1-(A+B+C),(dest1,pos2):A,(pos1,dest2):B,(dest1,dest2):C}
            return sol

    
    def rewards(self, state, actions): # recompenses immédiates {0:r1,1:r2}
        pos1, pos2 = state
        dest1 = self.next_position(pos1, actions[0])
        dest2 = self.next_position(pos2, actions[1])

        #goal
        if dest1 == self.goal_position and dest2 == self.goal_position : #but joueurs 1 et 2 simultannément
            return {0:100, 1:100}
        if dest1 == self.goal_position : #but joueur 1
            return {0:100, 1:0}
        if dest2 == self.goal_position : #but joueur 2
            return {0:0, 1:100}
        #run into each other
        if dest1 == dest2 : 
            return {0:-1, 1:-1}
        #otherwise
        return {0:0, 1:0}