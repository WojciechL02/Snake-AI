import pygame
from food import Food
from snake import Snake
from decision_tree.prepare_data import create_dataset
from naive_bayes.prepare_data import create_bayes_dataset
from agents import (MLPAgent,
                    HumanAgent,
                    DecisionTreeAgent,
                    NaiveBayesAgent,
                    RandomForestAgent,
                    SVMAgent)


def main():
    pygame.init()
    bounds = (300, 300)
    window = pygame.display.set_mode(bounds)
    pygame.display.set_caption("Snake")

    block_size = 30
    snake = Snake(block_size, bounds)
    food = Food(block_size, bounds, lifetime=100)

    # agent = HumanAgent(block_size, bounds)

    # dataset = create_dataset("data")  # Compulsory for RF, DT, SVM !
    # agent = DecisionTreeAgent(dataset)
    # agent = RandomForestAgent(dataset)
    # agent = SVMAgent(dataset)

    # dataset = create_bayes_dataset("data")  # Compulsory for Naive Bayes !
    # agent = NaiveBayesAgent(dataset)

    path = "18_0.03_5_128_.pth"
    agent = MLPAgent(path)

    scores = []
    run = True
    pygame.time.delay(1000)
    while run:
        pygame.time.delay(80)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        game_state = {"food": (food.x, food.y),
                      "snake_body": snake.body,
                      "snake_direction": snake.direction}

        direction = agent.act(game_state)
        snake.turn(direction)

        snake.move()
        snake.check_for_food(food)
        food.update()

        if snake.is_wall_collision() or snake.is_tail_collision():
            pygame.display.update()
            pygame.time.delay(300)
            scores.append(snake.length - 3)
            snake.respawn()
            food.respawn()

        window.fill((0, 0, 0))
        snake.draw(pygame, window)
        food.draw(pygame, window)
        pygame.display.update()

    print(f"Scores: {scores}")
    agent.dump_data()
    pygame.quit()


if __name__ == "__main__":
    main()
