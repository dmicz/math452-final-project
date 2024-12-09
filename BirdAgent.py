import numpy as np
from multiprocessing import shared_memory, Lock 
import time
import random


class Bird_Agent:
    def __init__(self, state_queue, action_queue, num_agents):
        self.state_queue = state_queue
        self.action_queue = action_queue
        self.state = []
        self.num_agents = num_agents
    
    def agent_task(self):
        while True:
            if not self.state_queue.empty():
                self.state = self.state_queue.get()  # Get the current state from the queue
                #get action for the current state
                self.get_action()
                print("Agent sees state:", self.state)
                print("Live agents: ", len(self.state))
    
    #call the nn, send the state and get the action, put the recieved action in the queue
    def get_action(self):
        
        action = np.random.randint(0, 2, size=self.num_agents)
        if not self.action_queue.empty():
            self.action_queue.get()
        self.action_queue.put(action)

                

    
    

