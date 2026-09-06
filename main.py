import numpy as np
import torch
from environment import Environment
from fish import Fish
from nn_model import SimpleRNN
from visualize import Visualizer

config = {
    "width": 1000,
    "height": 600,
    "dt": 0.01,           # 10ms
    "food_spawn_rate": 0.02,  # 毎tickの出現確率（1秒あたり2個）
    "max_food": 80,       # 餌の上限
}

vis = Visualizer(config["width"], config["height"])

env = Environment(config)
nn = SimpleRNN()
fish = Fish(config, nn)
env.fish_list.append(fish)

# 餌を初期配置
env.add_food_random(30)

for step in range(20000):  # 200秒
    # 餌の出現（上限を設ける）
    if np.random.rand() < config["food_spawn_rate"] and len(env.food_list) < config["max_food"]:
        env.add_food_random(1)

    env.step(config["dt"])

    if step % 100 == 0:
        # 魚がまだ生存しているか確認
        if env.fish_list:
            f = env.fish_list[0]
            print(f"t={step*config['dt']:.1f}s, energy={f.energy:.1f}, food={len(env.food_list)}, alive={f.alive}")
        else:
            print(f"t={step*config['dt']:.1f}s, 魚は死亡しました")
            break

    # もし魚が死んだら新しい魚を投入（観察用）
    if not env.fish_list and step < 19000:
        print("魚が死んだため、新しい魚を投入します")
        nn = SimpleRNN()  # 新しいNN
        fish = Fish(config, nn)
        env.fish_list.append(fish)

    if step % 100 == 0:
        f = env.fish_list[0]
        speed = np.hypot(f.vx, f.vy)
        print(f"t={step*config['dt']:.1f}s, pos=({f.x:.0f},{f.y:.0f}), speed={speed:.2f}, turn={f.turn:.2f}, accel={f.accel:.2f}")

    if not vis.render(env):
        break
