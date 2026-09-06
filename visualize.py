# visualize.py
import pygame
import numpy as np

class Visualizer:
    def __init__(self, width, height):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()

    def render(self, env):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        self.screen.fill((0, 100, 200))  # 水色
        # 餌
        for food in env.food_list:
            pygame.draw.circle(self.screen, (255, 0, 0), (int(food["x"]), int(food["y"])), 3)
        # 魚
        for fish in env.fish_list:
            pygame.draw.circle(self.screen, (255, 255, 0), (int(fish.x), int(fish.y)), 6)
        pygame.display.flip()
        self.clock.tick(60)  # 60FPS
        return True
