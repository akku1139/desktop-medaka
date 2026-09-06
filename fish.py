import numpy as np
import torch

class Fish:
    def __init__(self, config, nn):
        self.x = np.random.rand() * config["width"]
        self.y = np.random.rand() * config["height"]
        self.vx = 0.0
        self.vy = 0.0
        self.angle = np.random.rand() * 2 * np.pi
        self.energy = 300.0
        self.age = 0
        self.nn = nn
        self.inference_interval = 0.2  # 10Hz
        self.time_since_inference = 0.0
        self.hidden_state = torch.zeros(1, 16)
        self.turn = 0.0
        self.accel = 0.0
        self.alive = True

    def sense(self, env):
        # 壁までの距離（4方向）
        wall_left = self.x / env.width
        wall_right = (env.width - self.x) / env.width
        wall_top = self.y / env.height
        wall_bottom = (env.height - self.y) / env.height

        # 最も近い餌
        if env.food_list:
            distances = [np.hypot(f["x"]-self.x, f["y"]-self.y) for f in env.food_list]
            idx = np.argmin(distances)
            food_dx = (env.food_list[idx]["x"] - self.x) / env.width
            food_dy = (env.food_list[idx]["y"] - self.y) / env.height
            food_dist = distances[idx] / np.hypot(env.width, env.height)
            has_food = 1.0
        else:
            food_dx = 0.0
            food_dy = 0.0
            food_dist = 1.0
            has_food = 0.0

        # 自分の状態
        energy_norm = self.energy / 200.0
        speed = np.hypot(self.vx, self.vy) / 100.0

        obs = torch.tensor([
            wall_left, wall_right, wall_top, wall_bottom,
            food_dx, food_dy, food_dist, has_food,
            energy_norm, speed,
            np.cos(self.angle), np.sin(self.angle)
        ], dtype=torch.float32)
        return obs

    def update(self, env, dt):
        self.age += dt
        # 推論
        self.time_since_inference += dt
        if self.time_since_inference >= self.inference_interval:
            obs = self.sense(env)
            with torch.no_grad():
                out, self.hidden_state = self.nn(obs.unsqueeze(0), self.hidden_state)
                self.turn = torch.tanh(out[0,0]).item()
                self.accel = torch.tanh(out[0,1]).item()
            self.time_since_inference = 0.0
            self.energy -= 0.01  # 推論コスト

        # 運動
        self.angle += self.turn * 3.0 * dt
        speed = np.hypot(self.vx, self.vy)
        if self.accel > 0:
            self.vx += np.cos(self.angle) * 50.0 * dt
            self.vy += np.sin(self.angle) * 50.0 * dt
        else:
            # 摩擦
            self.vx *= 0.98
            self.vy *= 0.98

        # 位置更新
        self.x += self.vx * dt
        self.y += self.vy * dt
        # 壁で反射 + 向きを更新
        if self.x < 0:
            self.x = 0
            self.vx *= -0.5
            self.angle = np.arctan2(self.vy, self.vx) + np.random.uniform(-0.5, 0.5)
        if self.x > env.width:
            self.x = env.width
            self.vx *= -0.5
            self.angle = np.arctan2(self.vy, self.vx) + np.random.uniform(-0.5, 0.5)
        if self.y < 0:
            self.y = 0
            self.vy *= -0.5
            self.angle = np.arctan2(self.vy, self.vx) + np.random.uniform(-0.5, 0.5)
        if self.y > env.height:
            self.y = env.height
            self.vy *= -0.5
            self.angle = np.arctan2(self.vy, self.vx) + np.random.uniform(-0.5, 0.5)

        # エネルギー消費（基礎代謝＋運動）
        self.energy -= 0.1 * dt
        self.energy -= 0.1 * speed * dt

        # 餌を食べる（リストコピーで安全に）
        for food in env.food_list[:]:
            if np.hypot(food["x"]-self.x, food["y"]-self.y) < 10:
                self.energy += food["energy"]
                env.food_list.remove(food)
                break

        if self.energy <= 0:
            self.alive = False
