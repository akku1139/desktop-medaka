import numpy as np
import torch

class Environment:
    def __init__(self, config):
        self.width = config["width"]
        self.height = config["height"]
        self.fish_list = []
        self.food_list = []
        self.time = 0

    def add_food_random(self, n):
            for _ in range(n):
                self.food_list.append({
                    "x": np.random.rand() * self.width,
                    "y": np.random.rand() * self.height,
                    "energy": 30.0,
                    "age": 0,
                    "max_age": 600,  # ticks (10秒)
                })

    def step(self, dt):
        self.time += dt
        # 魚の更新
        for fish in self.fish_list:
            fish.update(self, dt)
        # 餌の更新・摂食
        self.update_food(dt)
        # 死亡判定
        self.remove_dead()

    def update_food(self, dt):
        # 餌の寿命管理と摂食
        for food in self.food_list:
            food["age"] += dt
        self.food_list = [f for f in self.food_list if f["age"] < f["max_age"]]

    def remove_dead(self):
        self.fish_list = [f for f in self.fish_list if f.alive]
