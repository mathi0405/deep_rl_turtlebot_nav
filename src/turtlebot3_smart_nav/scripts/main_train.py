#!/usr/bin/env python3
import rospy
import yaml
import torch
import torch.optim as optim
import random
import os
from collections import deque

from env_bridge import TurtleBot3NavigationEnv
from agent_network import TurtleBotDQN

def main():
    rospy.init_node('drl_training_node', anonymous=True)
    
    config_path = os.path.join(os.path.dirname(__file__), '../config/training_params.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    lr = config['agent_config']['learning_rate']
    
    env = TurtleBot3NavigationEnv()
    policy_net = TurtleBotDQN()
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    memory_buffer = deque(maxlen=config['agent_config']['experience_replay_size'])
    
    rospy.loginfo("Initialization complete. Starting DRL training loop...")
    
    for episode in range(config['environment_config']['maximum_steps_per_episode']):
        state = env.reset()
        
        # Add padding to make state 26-dimensional (24 Lidar + 2 Goal metrics)
        padded_state = np.append(state, [1.0, 0.0]) 
        state_tensor = torch.FloatTensor(padded_state).unsqueeze(0)
        done = False
        total_reward = 0
        
        while not done and not rospy.is_shutdown():
            if random.random() < 0.1:
                action = random.randint(0, 4)
            else:
                with torch.no_grad():
                    action = policy_net(state_tensor).argmax().item()
                    
            next_state, reward, done = env.step(action)
            total_reward += reward
            
            padded_next = np.append(next_state, [1.0, 0.0])
            memory_buffer.append((padded_state, action, reward, padded_next, done))
            
            state_tensor = torch.FloatTensor(padded_next).unsqueeze(0)
            padded_state = padded_next
            
        rospy.loginfo(f"Episode {episode} | Total Reward: {total_reward:.2f}")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
