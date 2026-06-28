#!/usr/bin/env python3
import rospy
import yaml
import torch
import numpy as np
import os
from collections import deque
from env_bridge import TurtleBot3NavigationEnv

from algorithms import TurtleBotDQN, DDPGActor, DDPGCritic, TD3Actor, TD3Critic, SACActor, SACCritic, CrossQActor, CrossQCritic

def get_agent(algo_name):
    if algo_name == "DQN": return TurtleBotDQN(), None
    elif algo_name == "DDPG": return DDPGActor(), DDPGCritic()
    elif algo_name == "TD3": return TD3Actor(), TD3Critic()
    elif algo_name == "SAC": return SACActor(), SACCritic()
    elif algo_name == "CrossQ": return CrossQActor(), CrossQCritic()
    else: raise ValueError(f"Unknown Algorithm: {algo_name}")

def main():
    rospy.init_node('rl_train_node', anonymous=True)
    config_path = os.path.join(os.path.dirname(__file__), '../config/training_params.yaml')
    with open(config_path, 'r') as f: config = yaml.safe_load(f)
        
    algo_name = config['agent_config']['active_algorithm']
    rospy.loginfo(f"--- STARTING TRAINING FRAMEWORK: {algo_name} ---")
    
    env = TurtleBot3NavigationEnv()
    actor, critic = get_agent(algo_name)
    memory = deque(maxlen=config['agent_config']['experience_replay_size'])
    
    for episode in range(config['environment_config']['maximum_steps_per_episode']):
        state = np.append(env.reset(), [1.0, 0.0])
        done = False
        total_reward = 0
        
        while not done and not rospy.is_shutdown():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Action Selection Block
            with torch.no_grad():
                if algo_name == "DQN":
                    action = actor(state_tensor).argmax().item() # Discrete
                else:
                    action = actor(state_tensor).squeeze().numpy() # Continuous
                    
            next_state_raw, reward, done = env.step(action)
            next_state = np.append(next_state_raw, [1.0, 0.0])
            total_reward += reward
            
            memory.append((state, action, reward, next_state, done))
            state = next_state
            
            # TODO: Team inserts specific optimization backprop rules (Loss functions) here.
            
        rospy.loginfo(f"Episode {episode} | Reward: {total_reward:.2f}")

if __name__ == '__main__':
    try: main()
    except rospy.ROSInterruptException: pass
