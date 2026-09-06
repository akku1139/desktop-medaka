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
        self.inference_interval = 0.2  # 5Hz
        self.time_since_inference = 0.0
        self.hidden_state = torch.zeros(1, 16)
        self.turn = 0.0
        self.accel = 0.0
        self.alive = True

        # 学習関連
        self.learning_rate = 0.01          # 学習率
        self.plasticity_cost_factor = 0.001 # 可塑性コスト係数
        self.elig_ih = torch.zeros_like(nn.rnn.weight_ih)  # 入力→隠れ層の適格度跡
        self.elig_hh = torch.zeros_like(nn.rnn.weight_hh)  # 隠れ層→隠れ層の適格度跡
        self.elig_decay = 0.9              # 適格度跡の減衰率
        self.prev_hidden = None            # 前回の隠れ状態（学習用）
        self.last_obs = None               # 前回の入力（学習用）

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
            # 前回の隠れ状態を保存（学習用）
            prev_hidden = self.hidden_state.clone()
            with torch.no_grad():
                out, self.hidden_state = self.nn(obs.unsqueeze(0), self.hidden_state)
                self.turn = torch.tanh(out[0,0]).item()
                self.accel = torch.tanh(out[0,1]).item()
            self.time_since_inference = 0.0
            self.energy -= 0.01  # 推論コスト

            # 適格度跡の更新（ヘブ学習の準備）
            with torch.no_grad():
                x = obs.unsqueeze(0)           # (1, input_size)
                h_prev = prev_hidden            # (1, hidden_size)
                h_curr = self.hidden_state      # (1, hidden_size)

                # 外積で適格度を計算
                # 入力→隠れ層: x^T * h_curr
                elig_ih_new = torch.mm(h_curr.t(), x)   # (hidden_size, input_size)
                # 隠れ層→隠れ層: h_prev^T * h_curr
                elig_hh_new = torch.mm(h_prev.t(), h_curr)  # (hidden_size, hidden_size)

                # 適格度跡の更新（指数移動平均）
                self.elig_ih = self.elig_decay * self.elig_ih + elig_ih_new
                self.elig_hh = self.elig_decay * self.elig_hh + elig_hh_new

            # 学習用に前回の状態を保存
            self.last_obs = obs
            self.prev_hidden = prev_hidden

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

        # 餌を食べる
        ate_food = False
        for food in env.food_list[:]:
            if np.hypot(food["x"]-self.x, food["y"]-self.y) < 10:
                self.energy += food["energy"]
                env.food_list.remove(food)
                ate_food = True
                break

        # 報酬が得られたら学習を実行
        if ate_food:
            reward = 20.0  # 餌のエネルギー量（報酬信号）
            with torch.no_grad():
                # 適格度跡に報酬を掛けて重み更新量を計算
                delta_w_ih = self.learning_rate * reward * self.elig_ih
                delta_w_hh = self.learning_rate * reward * self.elig_hh

                # 重みを更新
                self.nn.rnn.weight_ih.add_(delta_w_ih)
                self.nn.rnn.weight_hh.add_(delta_w_hh)

                # 可塑性コスト（重み変化量の絶対値和）
                plasticity_cost = self.plasticity_cost_factor * (
                    torch.sum(torch.abs(delta_w_ih)).item() +
                    torch.sum(torch.abs(delta_w_hh)).item()
                )
                self.energy -= plasticity_cost

                # 適格度跡をリセット
                self.elig_ih.zero_()
                self.elig_hh.zero_()

        if self.energy <= 0:
            self.alive = False
