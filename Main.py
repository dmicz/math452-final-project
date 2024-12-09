from multiprocessing import Process, Queue
from FlappyBird import FlappyBirdGame
from BirdAgent import Bird_Agent
import numpy as np
import sys  
import traceback


num_agents = 10
episodes = 2

if __name__ == "__main__":   
        action_queue = Queue()
        state_queue = Queue()

        #define the state of the game
        game = FlappyBirdGame(frames=120, state_queue=state_queue, action_queue= action_queue, episodes=episodes, num_agents= num_agents)
        game.start_game()
        bird = Bird_Agent(state_queue=state_queue, action_queue= action_queue, num_agents= num_agents)

   
        agent_process = Process(target=bird.agent_task)
        agent_process.start()

        try:
            game.play_game()
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"An unexpected error  occurred: {e}")
            traceback.print_exc()
        finally:
            print('fin')
            while not state_queue.empty():
                state_queue.get_nowait()
            while not action_queue.empty():
                action_queue.get_nowait()
            agent_process.terminate()
            
            
            print(action_queue)
            print(state_queue)
            state_queue.close()
            state_queue.join_thread()
            action_queue.close()
            action_queue.join_thread()
            print('Reached')
            sys.exit()