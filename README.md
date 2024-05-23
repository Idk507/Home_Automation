# Home Automation System with PPO-based Learning

## Overview

This Python script simulates a home automation system using a Proximal Policy Optimization (PPO) agent for learning control policies. The home automation system includes various devices such as switches, fans, curtains, an air conditioner, and a TV. The PPO agent learns to control these devices based on the current state of the environment.

## Modules

### 1. PPOAgent

- **Properties:**
  - `state_dim`: Dimensionality of the state space.
  - `action_dim`: Dimensionality of the action space.
  - `learning_rate`: Learning rate for the optimizer.
  - `gamma`: Discount factor for future rewards.
  - `clip_ratio`: Clipping ratio for PPO loss.
  - `policy`: Neural network model representing the policy.
  - `optimizer`: Optimizer used for training.

- **Methods:**
  - `__init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.95, clip_ratio=0.2)`: Initializes the PPOAgent.
  - `build_policy_network(self)`: Builds the neural network model for the policy.
  - `get_action_prob(self, state)`: Computes the probability distribution over actions given a state.
  - `train(self, states, actions, advantages)`: Trains the PPO agent using the provided states, actions, and advantages.

### 2. HomeAutomationSystem

- **Properties:**
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

- **Methods:**
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

### 3. calculate_advantages

- **Function:**
  - `calculate_advantages(rewards, values, gamma=0.95)`: Calculates advantages for PPO training.
  - **Parameters:**
    - `rewards`: Array of rewards received at each time step.
    - `values`: Array of estimated values of states.
    - `gamma`: Discount factor for future rewards.
  - **Returns:**
    - Array of advantages for each time step.

## Execution

- The main block of the script initializes the encryption key and creates an instance of the HomeAutomationSystem.

- The system is then run for a specified number of episodes using the `run_home_automation` method.

## Dependencies

- The script requires the following dependencies:
  - `numpy`
  - `tensorflow`
  - `cryptography`

## Usage

1. Ensure the required dependencies are installed by executing this command:
    ```bash
    pip install -r requirements.txt
    ```  
2. Run the script:

    ```bash
    python home_automation_script.py
    ```

## Notes

- Adjust the sleep duration in the `run_home_automation` method based on your requirements.

- The script periodically saves episodes in encrypted CSV and JSON files.
