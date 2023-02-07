import matplotlib.pyplot as plt
import pygame
import torch

from food import Food
from snake import Snake
from qlearning_agent import QLearningAgent


def main():
    pygame.init()
    bounds = (300, 300)
    window = pygame.display.set_mode(bounds)
    pygame.display.set_caption("Snake")

    block_size = 30
    snake = Snake(block_size, bounds)
    food = Food(block_size, bounds)

    path = "2022_12_30_17_43_00_0.1_0.9_0.1.pickle"

    agent = QLearningAgent(block_size, bounds, 0.2, 0.9, 0.1, False, path)
    scores = []
    run = True
    pygame.time.delay(1000)
    reward, is_terminal = 0, False
    episode = 0
    total_episodes = 100
    while episode < total_episodes and run:
        pygame.time.delay(80)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        game_state = {"food": (food.x, food.y),
                      "snake_body": snake.body,
                      "snake_direction": snake.direction}

        direction = agent.act(game_state, reward, is_terminal)
        reward = -0.001
        is_terminal = False
        snake.turn(direction)
        snake.move()
        reward += snake.check_for_food(food)
        food.update()

        if snake.is_wall_collision() or snake.is_tail_collision():
            pygame.display.update()
            pygame.time.delay(1)
            scores.append(snake.length - 3)
            snake.respawn()
            food.respawn()
            episode += 1
            reward -= 0.999
            is_terminal = True

        window.fill((0, 0, 0))
        snake.draw(pygame, window)
        food.draw(pygame, window)
        pygame.display.update()

    print(f"Scores: {scores}")
    print(f"Average: {sum(scores)/len(scores)}")
    print(f"Min: {min(scores)}")
    print(f"Max: {max(scores)}")
    if agent.is_training:
        scores = torch.tensor(scores, dtype=torch.float).unsqueeze(0)
        scores = torch.nn.functional.avg_pool1d(scores, 31, stride=1)
        plt.plot(scores.squeeze(0))
        plt.title(f"e={agent.epsilon}, d={agent.discount}, lr={agent.learning_rate}")
        plt.xlabel("Episode")
        plt.ylabel("Avg reward")
        filename = f"ms_{agent.epsilon}_{agent.discount}_{agent.learning_rate}"
        plt.savefig(f"plots/{filename}.png")
        print(f"Check out plots/{filename}.png")
        agent.dump_qfunction()
    pygame.quit()


if __name__ == "__main__":
    main()
