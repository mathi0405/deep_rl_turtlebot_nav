#!/usr/bin/env python3
import rospy
import torch
import numpy as np
import yaml
import os
from env_bridge import TurtleBot3NavigationEnv
from algorithms import TurtleBotDQN, DDPGActor, TD3Actor, SACActor, CrossQActor

def get_actor(algo_name):
    if algo_name == "DQN": return TurtleBotDQN()
    elif algo_name == "DDPG": return DDPGActor()
    elif algo_name == "TD3": return TD3Actor()
    elif algo_name == "SAC": return SACActor()
    elif algo_name == "CrossQ": return CrossQActor()

def main():
    rospy.init_node('rl_eval_node', anonymous=True)
    config_path = os.path.join(os.path.dirname(__file__), '../config/training_params.yaml')
    with open(config_path, 'r') as f: config = yaml.safe_load(f)
    
    algo_name = config['agent_config']['active_algorithm']
    env = TurtleBot3NavigationEnv()
    actor = get_actor(algo_name)
    
    weight_path = os.path.join(os.path.dirname(__file__), f'../models/{algo_name}_weights.pth')
    if os.path.exists(weight_path):
        actor.load_state_dict(torch.load(weight_path))
        actor.eval()
        rospy.loginfo(f"Loaded {algo_name} weights successfully.")
    else:
        rospy.logwarn(f"No weights found at {weight_path}. Running randomly.")
    
    for episode in range(10):
        state = np.append(env.reset(), [1.0, 0.0])
        done = False
        while not done and not rospy.is_shutdown():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                if algo_name == "DQN": action = actor(state_tensor).argmax().item()
                else: action = actor(state_tensor).squeeze().numpy()
            next_state_raw, _, done = env.step(action)
            state = np.append(next_state_raw, [1.0, 0.0])

if __name__ == '__main__':
    try: main()
    except rospy.ROSInterruptException: pass
