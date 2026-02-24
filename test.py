from benchmark.env_utils import configure_openenv_envs
configure_openenv_envs()

from envs.openspiel_env import OpenSpielAction, OpenSpielEnv
import asyncio
import random

async def play_random_game(num_games=10, num_steps=10):
    env = OpenSpielEnv(base_url="http://localhost:8000")
    print("Starting a random game of Blackjack...")
    try:
        for game in range(num_games):
            result = await env.reset()
            done = False
            step_num = 0

            while not done and step_num < num_steps:
                action = random.choice(result.observation.legal_actions)
                result = await env.step(OpenSpielAction(action_id=action, game_name="blackjack"))
                done = result.done
                step_num += 1
            
            print(f"Game ended with reward: {result.reward}")

    finally:
        await env.close()

if __name__ == "__main__":
    asyncio.run(play_random_game())