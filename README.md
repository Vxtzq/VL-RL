# A framework that turns any RL env into a LLM playable world
**DEMO:** Qwen 3.6 35B-A3B plays `SuperMarioBros-v0`
## Install
To run in same folder:
```
git clone https://github.com/Vxtzq/VL-RL/
cd VL-RL
```
Install with pip:
```
pip install vl-rl
```

## Minimal pipeline
Using gym_super_mario_bros as an example (can use any gym or gymnasium env)
```
import sys
import os
import json
import time


from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

from env import LLM_env
from agent import ollama_agent

FRAME_SKIP = 4


act_space_desc = """
0 - ['NOOP'], 1 - ['right'], 2 - ['right', 'A'], 3 - ['right', 'B'], 
4 - ['right', 'A', 'B'], 5 - ['A'], 6 - ['left'], 7 - ['left', 'A'], 
8 - ['left', 'B'], 9 - ['left', 'A', 'B'], 
10 - ['down'] (FORBIDDEN), 11 - ['up'] (FORBIDDEN)
- A = jump (hold to jump higher)
- B = run (hold while jumping = higher jump, useful for pipes)"""

goal = "Complete World 1-1 as fast as possible without dying."

mario_rules = """
# GAME-SPECIFIC RULES
- Brown mushrooms with feet = GOOMBAS (enemies, MUST jump over)
- Red/white spotted mushrooms = POWER-UPS (GOOD, collect them)
- Green pipes = solid walls (jump over with Action 4, NEVER try to enter)
- Green hills/bushes = BACKGROUND decoration, NOT obstacles (walk through them)
- Gaps/pits = instant death (MUST jump)
- ? blocks = hit from below to get items
- The FLAGPOLE at the end is the goal, NOT the pipes
- Running (Action 3) is optimal on flat ground with no enemies
- You MUST jump (Action 2 or 4) when a Goomba, pipe, or gap is ahead

# STRATEGY
- Run right (Action 3) when path is clear
- Jump (Action 2/4) over enemies and pipes
- Do NOT just run right blindly — look at the screen first
"""

full_prompt = prompt + mario_rules

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)
env, prompt = LLM_env(env, goal, act_space_desc, output_format="boxed")



done = True

info = {}


total_time = 0
step_times = []

for step in range(1000):
    if done:
        state = env.reset()

    obs = Image.fromarray(state)

    print(f"\n--- Step {step} ---")
    
    t0 = time.time()
    act, explain = ollama_agent(obs=obs, prompt=full_prompt + facts)
    elapsed = time.time() - t0
    
    step_times.append(elapsed)
    total_time += elapsed
    avg_time = total_time / len(step_times)
    
    print(f"⏱️  {elapsed:.1f}s (avg: {avg_time:.1f}s) | Action: {act} | {explain[:80]}")

    for i in range(FRAME_SKIP):
        state, reward, done, info = env.step(act)
        if done:
            break

    env.render()
    if done:
        print("==================================== Agent died ========================================")
        print(f"📊 Run stats: {len(step_times)} steps, total {total_time:.1f}s, avg {avg_time:.1f}s/step")

if step_times:
    print(f"\n{'='*60}")
    print(f"📊 FINAL STATS:")
    print(f"   Total steps: {len(step_times)}")
    print(f"   Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"   Avg per step: {avg_time:.1f}s")
    print(f"   Min: {min(step_times):.1f}s | Max: {max(step_times):.1f}s")
    print(f"{'='*60}")

env.close()
```
