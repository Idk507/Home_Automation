"""
Home Automation System with PPO-based Learning

This script controls a home automation system using a Proximal Policy Optimization (PPO) agent for learning control policies.
The home automation system includes various devices such as switches, fans, curtains, an air conditioner, and a TV.
The PPO agent learns to control these devices based on the current state of the environment.

Modules:
1. PPOAgent:
   - Represents the PPO agent responsible for learning control policies.
   - Utilizes a neural network to approximate the policy and implements training based on PPO algorithms.

2. HomeAutomationSystem:
   - Represents the home automation system.
   - Includes various devices and integrates the PPO agent for learning control policies.
   - Manages episode storage, encryption, and periodic saving of episodes.

Functions:
- `encrypt_data(data)`: Encrypts input data using AES encryption with the specified key.
- `compute_state_difference(current_state)`: Computes the difference between the current state and the previous state.
- `save_episodes()`: Saves episodes to encrypted CSV and JSON files.
- `get_current_time()`: Returns the current time.
- `get_external_temperature(season)`: Returns a random external temperature based on the season.
- `get_user_interaction()`: Simulates user interaction by randomly choosing True or False.
- `get_reward(user_override)`: Computes the reward based on user override.
- `update_home_automation_based_on_ppo()`: Updates the home automation system based on PPO agent's actions.
- `run_home_automation(num_episodes=1000)`: Runs the home automation system for a specified number of episodes.
- `calculate_advantages(rewards, values, gamma=0.95)`: Calculates advantages for PPO training.

Classes:
1. PPOAgent:
   - Properties:
     - `state_dim`: Dimensionality of the state space.
     - `action_dim`: Dimensionality of the action space.
     - `learning_rate`: Learning rate for the optimizer.
     - `gamma`: Discount factor for future rewards.
     - `clip_ratio`: Clipping ratio for PPO loss.
     - `policy`: Neural network model representing the policy.
     - `optimizer`: Optimizer used for training.

   - Methods:
     - `__init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.95, clip_ratio=0.2)`: Initializes the PPOAgent.
     - `build_policy_network(self)`: Builds the neural network model for the policy.
     - `get_action_prob(self, state)`: Computes the probability distribution over actions given a state.
     - `train(self, states, actions, advantages)`: Trains the PPO agent using the provided states, actions, and advantages.

2. HomeAutomationSystem:
   - Properties:
     - `switch_states`: List representing the states of switches.
     - `fan_states`: List representing the states of fans.
     - `curtain_states`: List representing the states of curtains.
     - `ac_temperature`: Current temperature setting of the air conditioner.
     - `tv_channel`: Current TV channel.
     - `tv_volume`: Current TV volume.
     - `ppo_agent`: Instance of the PPOAgent for learning control policies.
     - `max_episodes`: Maximum number of episodes to store in the episode buffer.
     - `episode_buffer`: A deque to store episodes.
     - `episode_counter`: Counter to keep track of the number of episodes.
     - `episode_save_interval`: Interval for saving episodes.
     - `csv_filename`: File name for the CSV file to store episodes.
     - `json_filename`: File name for the JSON file to store episodes.
     - `encryption_key`: Key used for encrypting sensitive episode data.
     - `previous_state`: Stores the previous state for computing state differences.
     - `previous_action`: Stores the previous action for tracking agent actions.

   - Methods:
     - `__init__(self, encryption_key)`: Initializes the HomeAutomationSystem.
     - `encrypt_data(self, data)`: Encrypts the input data using AES encryption.
     - `compute_state_difference(self, current_state)`: Computes the difference between the current state and the previous state.
     - `save_episodes(self)`: Saves episodes to encrypted CSV and JSON files.
     - `get_current_time(self)`: Returns the current time.
     - `get_external_temperature(self, season)`: Returns a random external temperature based on the season.
     - `get_user_interaction(self)`: Simulates user interaction by randomly choosing True or False.
     - `get_reward(self, user_override)`: Computes the reward based on user override.
     - `update_home_automation_based_on_ppo(self)`: Updates the home automation system based on PPO agent's actions.
     - `run_home_automation(self, num_episodes=1000)`: Runs the home automation system for a specified number of episodes.
     - `__del__(self)`: Destructor to print a message when the HomeAutomationSystem instance is deleted.

3. calculate_advantages(rewards, values, gamma=0.95):
   - Function to calculate advantages for PPO training.
   - Parameters:
     - `rewards`: Array of rewards received at each time step.
     - `values`: Array of estimated values of states.
     - `gamma`: Discount factor for future rewards.
   - Returns:
     - Array of advantages for each time step.
"""
from typing import List, Dict, Union, Any, Tuple
import numpy as np
import tensorflow as tf
import random
import datetime
import time
import json
import csv
from collections import deque
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import secrets

# Generate a random 16-byte key (128 bits)
encryption_key = secrets.token_bytes(16)


class PPOAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.95, clip_ratio=0.2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.policy = self.build_policy_network()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    def build_policy_network(self):  
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.state_dim,)),
            tf.keras.layers.Dense(self.action_dim, activation='softmax')
        ])
        return model

    def get_action_prob(self, state):
        return self.policy.predict(np.array([state]))[0]

    def train(self, states, actions, advantages):
        actions = tf.keras.utils.to_categorical(actions, self.action_dim)
        advantages = np.array(advantages)

        with tf.GradientTape() as tape:
            policy_prob = self.policy(states, training=True)
            chosen_prob = tf.reduce_sum(actions * policy_prob, axis=1)
            old_policy_prob = tf.stop_gradient(chosen_prob)

            ratio = tf.exp(tf.math.log(chosen_prob + 1e-10) - tf.math.log(old_policy_prob + 1e-10))
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            surrogate = -tf.minimum(ratio * advantages, clipped_ratio * advantages)
            loss = tf.reduce_mean(surrogate)

        gradients = tape.gradient(loss, self.policy.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))
 

class HomeAutomationSystem:
    def __init__(self, encryption_key):
        #assuming that user has 3 switches, 3 fans, 2 curtains, 1 AC, 1 TV
        self.switch_states = [0] * 12
        self.fan_states = [0] * 3
        self.curtain_states = [1, 1]
        self.ac_temperature = 25
        self.tv_channel = 0
        self.tv_volume = 0

        num_states = (
            len(self.switch_states) +
            len(self.fan_states) +
            len(self.curtain_states) +
            1 +  # AC temperature
            2 +  # TV channel, volume
            1    # External temperature
        )
        self.ppo_agent = PPOAgent(state_dim=num_states, action_dim=2)
        # Additional attributes for episode storage
        self.max_episodes = 4320
        self.episode_buffer = deque(maxlen=self.max_episodes)
        self.episode_counter = 0
        self.episode_save_interval = 10  # Save every 10 episodes
        self.csv_filename = "episodes.csv"
        self.json_filename = "episodes.json"

        # Save header to CSV
        with open(self.csv_filename, 'w', newline='') as csvfile:
            fieldnames = ["timestamp", "state", "action", "reward", "temperature", "state_difference", "encrypted_data"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

        # Initialize encryption parameters
        self.encryption_key = encryption_key

        # Store previous state and action data
        self.previous_state = None
        self.previous_action = None

    def encrypt_data(self, data):
        '''
        Library :Cryptography
        AES - Advanced Encryption Standard (using 128 bit key size)
        CFB - Cipher FeedBack Mode -Encryption function using the output of the previous block (using 16 bit initialization vector)
        
        '''
        cipher = Cipher(algorithms.AES(self.encryption_key), modes.CFB(b'\0' * 16), backend=default_backend())
        encryptor = cipher.encryptor()
        encrypted_data = encryptor.update(data.encode()) + encryptor.finalize()
        return encrypted_data

    def compute_state_difference(self, current_state):
        # Compute the difference between the current state and the previous state
        if self.previous_state is not None:
            state_difference = current_state - self.previous_state
        else:
            state_difference = np.zeros_like(current_state)
        return state_difference

    def save_episodes(self):
        # Save episodes to encrypted CSV and JSON files
        with open(self.csv_filename, 'a', newline='') as csvfile:
            fieldnames = ["timestamp", "state", "action", "reward", "temperature", "state_difference", "encrypted_data"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            for data_point in self.episode_buffer:
                # Convert data to JSON string
                json_data = json.dumps(data_point)

                # Encrypt the JSON data
                encrypted_json_data = self.encrypt_data(json_data)

                # Write the encrypted data to the CSV file
                writer.writerow({
                    "timestamp": str(data_point["timestamp"]),
                    "state": str(data_point["state"]),
                    "action": str(data_point["action"]),
                    "reward": str(data_point["reward"]),
                    "temperature": str(data_point["temperature"]),
                    "state_difference": str(data_point["state_difference"]),
                    "encrypted_data": str(encrypted_json_data)
                })

        # Save to encrypted JSON
        with open(self.json_filename, 'w') as json_file:
            encrypted_episode_data = self.encrypt_data(json.dumps(list(self.episode_buffer), indent=2))
            json_file.write(str(encrypted_episode_data))

        print(f"Episodes saved to {self.csv_filename} and {self.json_filename} (encrypted)")

    def get_current_time(self):
        return datetime.datetime.now().time()

    def get_external_temperature(self, season):
        #This is a dummy Function
        #[REPLACE THIS CODE WITH THE CODE TO GET THE DATA OF TEMPERATURE FROM THE TEMPERATURE SENSOR]
        if season == 'summer':
            return random.uniform(23, 42)
        elif season == 'winter':
            return random.uniform(10, 32)
        else:
            raise ValueError("Invalid season")

    def get_user_interaction(self,state_difference):
        #[user interaction can be known by the state difference]
        if any(state_difference):
            return True
        else:
            return False

    def get_reward(self, user_override):
        #[THIS FUNCTION NEEDS TO BE TUNED FOR MORE COMPLEX REWARD FUNCTIONS]
        return -1 if user_override else 1
    
    def get_switch_states(self):
        return self.switch_states
    
    def get_fan_states(self):
        return self.fan_states
    
    def get_curtain_states(self):
        return self.curtain_states
    
    def get_ac_temperature(self):
        return self.ac_temperature
    
    def get_tv(self):
        return self.tv_channel, self.tv_volume
    
    def update_home_automation_based_on_ppo(self):
        current_time = self.get_current_time()

        #[CURRENT STATES MUST BE FETCHED FROM THE REAL DEVICES]
        self.switch_states=self.get_switch_states()
        self.fan_states=self.get_fan_states()
        self.curtain_states=self.get_curtain_states()
        self.ac_temperature=self.get_ac_temperature()
        self.tv_channel, self.tv_volume = self.get_tv() 
        current_state = np.concatenate([
            np.array(self.switch_states),
            np.array(self.fan_states),
            np.array(self.curtain_states),
            np.array([self.ac_temperature]),
            np.array([self.tv_channel, self.tv_volume]),
            np.array([self.get_external_temperature('summer')])  # Add temperature to state
        ])

        # Fetch the current state before computing new results
        previous_state = self.previous_state
        previous_action = self.previous_action

        # Compute the state difference
        state_difference = self.compute_state_difference(current_state)
        print("State Difference: ",state_difference)

        user_override = self.get_user_interaction(state_difference)
        reward = self.get_reward(user_override)

        action_prob = self.ppo_agent.get_action_prob(current_state)
        action = np.random.choice(range(2), p=action_prob)

        states = np.array([current_state])
        actions = np.array([action])
        rewards = np.array([reward])

        values = np.zeros_like(rewards)
        advantages = calculate_advantages(rewards, values)
        self.ppo_agent.train(states, actions, advantages)

        timestamp = str(current_time)
        temperature = self.get_external_temperature('summer')
        episode_data = {
            "timestamp": timestamp,
            "state": current_state.tolist(),
            "action": actions.tolist(),
            "reward": reward,
            "temperature": temperature,
            "state_difference": state_difference.tolist(),
        }
        self.episode_buffer.append(episode_data)

        # Save episodes periodically
        if self.episode_counter % self.episode_save_interval == 0:
            self.save_episodes()

        # Update previous state and action
        self.previous_state = current_state
        self.previous_action = action

        # Increment episode counter
        self.episode_counter += 1

    def run_home_automation(self, num_episodes=1000):
        try:
            for episode in range(num_episodes):
                current_time = self.get_current_time()

                self.update_home_automation_based_on_ppo()

                # Print current state (for demonstration purposes)
                print("Episode:", episode)
                print("Switch States:", self.switch_states)
                print("Fan States:", self.fan_states)
                print("Curtain States:", self.curtain_states)
                print("AC Temperature:", self.ac_temperature)
                print("TV Channel, Volume:", self.tv_channel, self.tv_volume)
            
                # Sleep for a certain duration before the next iteration
                # Sleep for 5 minutes
                time.sleep(300)

        except KeyboardInterrupt:
            print("Training interrupted. Cleaning up resources.")
            self.save_episodes()

    def __del__(self):
        print("HomeAutomationSystem instance is being deleted.")

def calculate_advantages(rewards, values, gamma=0.95):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = running_add * gamma + rewards[t]
        advantages[t] = running_add - values[t]
    return advantages



if __name__ == "__main__":
    #print(f"Generated encryption key: {encryption_key}")
    encryption_key = encryption_key[:16]

    home_automation_system = HomeAutomationSystem(encryption_key)
    home_automation_system.run_home_automation()