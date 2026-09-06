import numpy as np
import torch
from environment import Environment
from fish import Fish
from nn_model import SimpleRNN

config = {
    "width": 1000,
    "height": 600,
    "dt": 0.01,
    "food_spawn_rate": 0.005,
    "max_food": 30,
}

def run_simulation(learning_enabled, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = Environment(config)

    num_fish = 5
    for i in range(num_fish):
        nn = SimpleRNN()
        fish = Fish(config, nn)
        fish.learning_rate = 0.01 if learning_enabled else 0.0
        env.fish_list.append(fish)

    env.add_food_random(15)

    survival_times = []
    for step in range(30000):  # 300秒
        if np.random.rand() < config["food_spawn_rate"] and len(env.food_list) < config["max_food"]:
            env.add_food_random(1)
        env.step(config["dt"])
        # 死亡個体の記録
        if len(env.fish_list) < num_fish:
            # 死亡時刻を記録（簡易的に現在時刻を記録）
            pass

        if not env.fish_list:
            break

    # 最終的な生存時間と個体数を返す
    return step * config["dt"], len(env.fish_list)

# 実験実行
learning_results = []
no_learning_results = []

for seed in range(10):
    t_survive, n_survive = run_simulation(True, seed)
    learning_results.append((t_survive, n_survive))
    t_survive, n_survive = run_simulation(False, seed)
    no_learning_results.append((t_survive, n_survive))

print("学習あり:", learning_results)
print("学習なし:", no_learning_results)
