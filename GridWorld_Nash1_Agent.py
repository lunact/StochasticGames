from non_RL_Agent import non_RL_Agent
from GridWorld import GridWorld
import numpy as np
import random

class GridWorld_Nash1_Agent(non_RL_Agent):
    """
    A Specific Agent class that plays a Nash equilibrium strategy (as opponent aka playerID=1) for GridWorld Game
    """

    def __init__(self,game,playerID):
        super(GridWorld_Nash1_Agent,self).__init__(game,playerID)

    def compute_policy(self):
        self.pi = {state: {action: 0 for action in self.g.actions(self.playerID, state)} for state in self.g.states()}
        for x in self.g.cells :
            if x != (2,1) :
                if x != (0,2) :
                    self.pi[(x,(0,2))]['Left'] = 1
                if x != (0,1) :
                    self.pi[(x,(0,1))]['Up'] = 1
                if x != (1,1) :
                    self.pi[(x,(1,1))]['Up'] = 1
        return self.pi