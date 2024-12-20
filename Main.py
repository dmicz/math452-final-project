from FlappyBird import FlappyBirdGame
from BirdAgent import Bird_Agent
import threading
import queue

num_agents = 50
episodes = 3000r

if __name__ == "__main__":
    input_size = 8  # Number of state features
    hidden_size = 128
    output_size = 2  # Number of actions (flap or not flap)

    action_queue = queue.Queue()
    state_queue = queue.Queue()

    # Initialize the game and start the game in the main thread
    game = FlappyBirdGame(frames=120, state_queue=state_queue, action_queue=action_queue, episodes=episodes, num_agents=num_agents)
    game.start_game()
    game.play_game()

    bird = Bird_Agent(state_queue=state_queue, action_queue=action_queue, num_agents=num_agents,
                      input_size=input_size, hidden_size=hidden_size, output_size=output_size)

    def agent_task():
        bird.agent_task()
        positive_states = []
        negative_states = []

        while True:
            if not state_queue.empty():
                states = state_queue.get()
                for state in states:
                    if state[6] > 0:
                        positive_states.append(state)
                    else:
                        negative_states.append(state)

            if positive_states and negative_states:
                bird.train(positive_states, negative_states)

            # Optionally, add a sleep to control the frequency of training
            # time.sleep(0.1)  # You can adjust this to control the training frequency

            # Clear the lists to avoid unnecessary memory usage
            positive_states.clear()
            negative_states.clear()

    # Start the agent logic in a separate thread
    agent_thread = threading.Thread(target=agent_task)
    agent_thread.start()

    try:
        for episode in range(episodes):
            while game.game_on():
                # Main game loop: play the game and collect states and actions
                pass  # The agent logic should handle the game play and training

    except KeyboardInterrupt:
        pass
    finally:
        agent_thread.join()
        # No need to close the queue (Queue objects in Python don't have a 'close' method)
        # If you still need to signal the end of the game, use a sentinel value
        action_queue.put(None)
        state_queue.put(None)


