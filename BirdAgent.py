import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ForwardForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ForwardForwardNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return x
    
    def goodness(self, x):
        return torch.sum(x**2, dim=1)
    
class BirdAgent:
    def __init__(self, state_queue, action_queue, num_agents, input_size, hidden_size, output_size, learning_rate=0.01, threshold=1.0, epsilon=0.1):
        self.state_queue = state_queue
        self.action_queue = action_queue
        self.num_agents = num_agents
        self.nn = ForwardForwardNN(input_size, hidden_size, output_size)
        self.optimizer = optim.Adam(self.nn.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.threshold = threshold
        self.epsilon = epsilon

    def agent_task(self):
        while True:
            states = self.state_queue.get()
            if states is None:
                break
            actions = self.get_action(states)
            self.action_queue.put(actions)

    def get_action(self, states):
        actions = []
        for state in states:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                goodness_scores = self.nn.goodness(self.nn(state))
            if np.random.rand() < self.epsilon:
                action = np.random.choice([0, 1])
            else:
                action = torch.argmax(goodness_scores).item()
            actions.append(action)
        return actions

    def train(self, positive_states, negative_states):
        self.optimizer.zero_grad()
        pos_states = torch.tensor(positive_states, dtype=torch.float32)
        pos_goodness = self.nn.goodness(self.nn(pos_states))
        pos_loss = torch.mean((pos_goodness - self.threshold)**2)
        neg_states = torch.tensor(negative_states, dtype=torch.float32)
        neg_goodness = self.nn.goodness(self.nn(neg_states))
        neg_loss = torch.mean(neg_goodness**2)
        loss = pos_loss + neg_loss
        loss.backward()
        self.optimizer.step()
        print(f"Training Loss: {loss.item():.4f} | Positive Loss: {pos_loss.item():.4f} | Negative Loss: {neg_loss.item():.4f}")
