# RocketLeague-PPO-Bot
 
A Rocket League bot trained from scratch using Proximal Policy Optimization (PPO) and a custom C++ reinforcement learning environment. Built on top of [GigaLearnCPP](https://github.com/ZealanL/GigaLearnCPP-Leak), with a custom reward system designed to produce competitive 1v1 gameplay.
 
## Training Progression

| 2 Billion Steps | 5 Billion Steps | 10 Billion Steps |
|----------------|----------------|-----------------|
| ![2B](Clips/2b.gif) | ![5B](Clips/5b.gif) | ![10B](Clips/10b.gif) |
| Early movement, chasing ball | Scoring goals, boost management | Aerials, demos, advanced mechanics |
 
## Overview
 
The bot learns to play Rocket League through self-play, receiving rewards for ball control, aerial maneuvers, boost management, and scoring goals. Training runs entirely in C++ for maximum performance, simulating hundreds of games in parallel on CPU or GPU.
 
After approximately 10 billion timesteps the bot reliably performs at a skill level outperforming approximately 80% of ranked Rocket League players.
 
## Architecture
 
| Component | Details |
|-----------|---------|
| Algorithm | PPO (Proximal Policy Optimization) |
| Network | Shared head + policy/critic branches, 3x256 layers, ReLU, LayerNorm |
| Optimizer | Adam (`lr = 1e-4`) |
| Observation | `AdvancedObs` — full game state per player |
| Action space | `DefaultAction` — discrete action parser |
| Simulation | RocketSim physics engine |
| Parallelism | 256 simultaneous games |
 
## Reward System
 
Rewards are weighted and combined at each timestep to shape behavior across multiple skill dimensions:
 
| Reward | Weight | Purpose |
|--------|--------|---------|
| `GoalReward` | 150 | Primary objective — score, penalize conceding |
| `DemoReward` (zero-sum) | 80 | Reward demolishing opponents |
| `AerialTouchReward` | 35 | Encourage aerial ball contacts |
| `TouchAccelReward` | 20 | Reward accelerating the ball on touch |
| `BumpReward` (zero-sum) | 20 | Reward bumping opponents |
| `PickupBoostReward` | 15 | Reward collecting boost |
| `VelocityBallToGoalReward` (zero-sum) | 2.5 | Push ball toward opponent goal |
| `SpeedReward` | 0.4 | General car speed |
| `SaveBoostReward` | 0.2 | Penalize wasting boost |
| `FaceBallReward` | 0.25 | Keep car oriented toward ball |
| `VelocityPlayerToBallReward` | 0.25 | Drive toward the ball |
| `AirReward` | 0.25 | Encourage time in the air |
| `WavedashReward` | 10 | Reward wavedash mechanics |
 
Zero-sum rewards are computed relative to the opposing team, preventing reward hacking through passive play.
 
## Training Configuration
 
```cpp
cfg.numGames = 256;          // Parallel game instances
cfg.tickSkip = 8;            // Physics ticks per action
cfg.ppo.tsPerItr = 100'000;  // Timesteps per PPO update
cfg.ppo.epochs = 2;          // PPO epochs per iteration
cfg.ppo.entropyScale = 0.035f;
cfg.ppo.gaeGamma = 0.99;     // Reward decay
cfg.tsPerSave = 50'000'000;  // Checkpoint every 50M steps
```
 
## State Initialization
 
Each episode starts in one of two states, sampled randomly:
 
- **Kickoff** (60%) - standard kickoff positions
- **Random** (40%) - random car and ball positions, velocities, and rotations
This mix ensures the bot learns both structured play and recovery from chaotic situations.
 
## Metrics Tracked
 
- In-air ratio
- Ball touch ratio
- Demo ratio
- Average car speed & speed towards ball
- Boost level
- Touch height (aerial quality)
- Goal speed
## Dependencies
 
- [RLGymCPP](https://github.com/ZealanL/RLGymPPO_CPP) - Rocket League gym environment in C++
- [GigaLearnCPP](https://github.com/ZealanL/GigaLearnCPP-Leak) - PPO learner framework
- [RocketSim](https://github.com/ZealanL/RocketSim) - Fast Rocket League physics simulation
- Collision meshes extracted from Rocket League (not included)
