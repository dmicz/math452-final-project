'''
    Required packages:
    pygame. can be installed through pip install pygame
    torch. best to install version equipped with cuda. can be downloaded at https://pytorch.org/get-started/locally/
    if possible utilize a gpu due to faster and better computation with reduced lag.
'''


import pygame
import random
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import traceback
import sys
from collections import deque
import time
import torch.optim as optim



class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 256)
        self.layer4 = nn.Linear(256, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        return self.layer3(x)


mp.set_start_method('spawn', force=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
print(device)
num_agents = 1
#episodes = itertools.count()
episodes = 1500
frames = 60
score_record = []
gamma = 0.9
buffer_capacity = 100_000
learning_rate = 1e-4
policy_net = DQN(6, 2).to(device)
target_net = DQN(6, 2).to(device)



target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
replay_buffer = deque(maxlen=100_000)
batch_size = 100
collision_on = True




    


class FlappyBirdGame:
    
    def __init__(self, frames, episodes = 1, num_agents = 2):
        self.screen = None
        self.clock = None
        self.game_font = None
        self.game_start = False
        self.score = 0.0
        self.frames = frames
        self.episodes = episodes
        self.num_agents = num_agents
        self.current_state = []
        self.birds = [{'rect': None, 'movement' : 0, 'score' : 0.0, 'active': True} for _ in range(num_agents)]
        
    def game_close(self):
        pygame.quit()

    def game_on(self):
        return self.game_start

    def draw_floor(self, floor_surface, floor_x_pos):
        self.screen.blit(floor_surface, (floor_x_pos, 900))
        self.screen.blit(floor_surface, (floor_x_pos + 576, 900))

    def create_pipe(self, pipe_surface, pipe_height):
        difference = 300
        random_pipe_pos = random.choice(pipe_height)
        bottom_pipe = pipe_surface.get_rect(midtop = (700, random_pipe_pos))
        top_pipe = pipe_surface.get_rect(midbottom = (700, random_pipe_pos - difference))
        return bottom_pipe, top_pipe

    def move_pipes(self, pipes):
        for pipe in pipes:
            pipe.centerx -= 500/self.frames 
        return pipes

    def draw_pipes(self, pipe_surface, pipes):
        for pipe in pipes:
            if pipe.bottom >= 1024:
                self.screen.blit(pipe_surface, pipe)
            else:
                flip_pipe = pygame.transform.flip(pipe_surface, False, True)
                self.screen.blit(flip_pipe, pipe)

    def check_collison(self, bird_rect, pipes):
        for pipe in pipes:
            if bird_rect.colliderect(pipe):
                return collision_on
        if bird_rect.top <= -100 or bird_rect.bottom >= 900:
            return True
        
        return False

    #display the score
    def score_display(self, game_mode):
        game_font = pygame.font.Font(None, 40)
        if game_mode == 'main_game':
            score_surface = game_font.render(str(int(self.birds[0]['score'])), True, (0, 0, 0))
            score_rect = score_surface.get_rect(center = (288, 100))
            self.screen.blit(score_surface, score_rect)

        if game_mode == 'game_over':
            score_surface = game_font.render(str(int(self.score)), True, (0, 0, 0))
            score_rect = score_surface.get_rect(center = (288, 100))
            self.screen.blit(score_surface, score_rect)

            high_score_surface = game_font.render(f'Highscore: ' + str(int(self.high_score)), True, (0, 0, 0))
            high_score_rect = high_score_surface.get_rect(center = (288, 150))
            self.screen.blit(high_score_surface, high_score_rect)

    #start game
    def start_game(self):
        pygame.init()
        self.screen = pygame.display.set_mode((576, 1024))
        self.clock = pygame.time.Clock()
        self.game_start = True
    
    #get the action for the specific state
    def get_action(self, epsilon):
        action = []
        for state in self.current_state:
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
            if np.random.random() < epsilon:
                action.append(1 if np.random.randint(0,17)==1 else 0)
            else:
                #recieve the Q-values from the DNN and output the action with the max q-value
                with torch.no_grad():
                    q_values = policy_net(state_tensor[:6])
                    action.append(q_values.argmax(dim=0).cpu().numpy())
        return action
    
    #get the state of the bird
    def get_state(self, prev_bird_position, bird_y, pipe_list, score, agent_num):
        bottom_pipe_pos = 0
        top_pipe_pos = 0
        distance_bird_pipe = 0
        bird_velocity = bird_y - prev_bird_position

        if len(pipe_list)!=0:
            bottom_pipe_pos= pipe_list[0].y
            top_pipe_pos =  pipe_list[1].y
            distance_bird_pipe = pipe_list[0].x - 100
            
        if distance_bird_pipe<0:
            score +=10
        state = [bird_y, bird_velocity, bottom_pipe_pos, top_pipe_pos, distance_bird_pipe, score, agent_num]
        
        return state
   
    def play_game(self):
        
        random.seed(90)
        epsilon = .7
        for i in range(episodes):
            start_time = time.time()
            #game variables
            print(f'Episode Number {i}')
            gravity = 0.25
            bird_movement = 0
            game_active = True
            #take actions every n frames
            current_frame = 0

            #init the background
            bg_surface = pygame.image.load('FlappyBirdDQN/assets/background-day.png').convert()
            bg_surface = pygame.transform.scale2x(bg_surface)

            #init the floor surface
            floor_surface = pygame.image.load('FlappyBirdDQN/assets/base.png').convert()
            floor_surface = pygame.transform.scale2x(floor_surface)
            floor_x_pos = 0

            #init the bird
            bird_surface = pygame.image.load('FlappyBirdDQN/assets/bluebird-midflap.png').convert_alpha()
            bird_surface = pygame.transform.scale2x(bird_surface)
            for bird in self.birds:
                bird["rect"] = bird_surface.get_rect(center = (100, 512))
            

            #init the pipes and store them in a list
            pipe_surface = pygame.image.load('FlappyBirdDQN/assets/pipe-green.png')
            pipe_surface = pygame.transform.scale2x(pipe_surface)
            pipe_list = []

            #record the previous position
            
            bird_position = []
            
            SPAWNPIPE = pygame.USEREVENT
            pygame.time.set_timer(SPAWNPIPE, 1200)
            pipe_height = [600]
            step = 0
            start_time = time.time()
            done = [0 for _ in range(self.num_agents)]
            reward = [0 for _ in range(self.num_agents)]
            epsilon = max(0.01, epsilon*0.99)
            
            while game_active:
                for event in pygame.event.get():
                    #quit the game if the user exits the window
                    if event.type == pygame.QUIT:
                        pygame.quit() 
                    #if a button is pressed and it is a space, make the bird flap
                    if event.type == SPAWNPIPE:
                        pipe_list.extend(self.create_pipe(pipe_surface, pipe_height))

                #get current state of the birds
                current_state = []
                
                
                for i in range(len(self.birds)):
                    if len(bird_position)==0:
                        bird_position = [314 for _ in range(self.num_agents)]
                    bird = self.birds[i]
                    if bird['active']:
                        current_state.append(self.get_state(bird_position[i] ,bird['rect'].centery, pipe_list, bird['score'],i))
                        bird_position[i] = bird['rect'].centery
                        done[i] = 0
                        reward[i] = bird['score']
                        
                        if len(pipe_list)!=0:
                            if  pipe_list[0].centerx == 100:
                                reward[i] += 10
                        
                    else:
                        done[i] = 1
                self.current_state = current_state

                #getting action from the action_queue and posting it
                action = self.get_action(epsilon=epsilon)
                
                
                
                for i in range(self.num_agents):
                    bird = self.birds[i]
                    if bird['active'] and i<len(action):
                        if action[i] == 1:
                            bird['movement'] = -6
                
                #if the game is active, apply changes to the birds state and simulate the background environment
                for bird in self.birds: 
                    bird['movement'] += gravity
                    bird['rect'].centery += bird['movement']
                bird_movement += gravity
                
               
                #adding the surfaces and bird to the screen
                self.screen.blit(bg_surface, (0,0))
                bird_count = self.num_agents
                i = 0
                for bird in self.birds:
                    if bird['active']:
                        self.screen.blit(bird_surface, bird['rect'])
                        bird['score'] += 0.05
                        bird['active'] = not self.check_collison(bird['rect'], pipe_list)
                        if not bird['active']:
                            reward[i] = - 10 
                            done[i] = 1
                    else:
                        bird_count -=1
                    i += 1
                if bird_count==0:
                    game_active = False

                

                #Pipes
                pipe_list = self.move_pipes(pipe_list)
                self.draw_pipes(pipe_surface, pipe_list)
                self.score_display('main_game')
                    
                floor_x_pos -= 1
                self.draw_floor(floor_surface, floor_x_pos)
                if floor_x_pos <= -576:
                    floor_x_pos = 0
                
                
                next_state = []
                
                for i in range(len(self.birds)):
                    if len(bird_position)==0:
                        bird_position = [314 for _ in range(self.num_agents)]
                    bird = self.birds[i]
                    if bird['active']:
                        next_state.append(self.get_state(bird_position[i] ,bird['rect'].centery, pipe_list, bird['score'],i))
                        bird_position[i] = bird['rect'].centery
                    else:
                        next_state.append([0, 0, 0, 0, 0, -100])
                if len(pipe_list)>0:
                    if pipe_list[0].centerx<0:
                        pipe_list = pipe_list[2:]

                #send the experience to the replay buffer
                for i in range(len(current_state)):
                    experience = [current_state[i][:6], action[i], reward[i], next_state[i][:6] , done[i]]
                    
                    replay_buffer.append(experience)
                
                # Wait until there are enough samples
                if len(replay_buffer) > batch_size:
              
            
                    # Sample a batch from the replay buffer
                    batch = random.sample(replay_buffer, batch_size)

                    states, actions, rewards, next_states, dones = zip(*batch)
                    states = torch.tensor(states, dtype=torch.float32).to(device)
                    
                    actions = torch.tensor(np.array(actions), dtype=torch.int64).to(device)
                    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
                    next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
                    dones = torch.tensor(dones, dtype=torch.float32).to(device)
                    
                    # Calculate Q-values and target Q-values
                    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        max_next_q_values = target_net(next_states).max(dim=1)[0]
                        target_q_values = rewards + (1 - dones) * gamma * max_next_q_values

                    # Compute loss
                    loss = torch.nn.functional.mse_loss(q_values, target_q_values)
                    # Backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Update target network periodically
                    step += 1
                    if step % 10 == 0:
                        target_net.load_state_dict(policy_net.state_dict())
                
                #update the visual of the game
                pygame.display.update()
                self.clock.tick(self.frames)
                current_frame +=1

            #reset the game and calculate the max score of all the birds
            pipe_list.clear()
            max_score_bird = max(self.birds, key=lambda bird: bird['score'])
            print(max_score_bird['score'])
            score_record.append(max_score_bird['score'])
            self.birds = [{'rect': None, 'movement' : 0, 'score' : 0.0, 'active': True} for _ in range(self.num_agents)]
        
        
        

        # Plotting
        

        pygame.quit()


if __name__ == '__main__':
    #create game object and start
    game = FlappyBirdGame(frames=frames, episodes=episodes, num_agents= num_agents)
    game.start_game()
    
    #play the game
    try:
        game.play_game()
        print("Score record per episode: ", score_record)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        
        print(score_record)
        print(f"An unexpected error  occurred: {e}")
        traceback.print_exc()
    finally:
        print('fin') 
        print('Reached')
        sys.exit()

