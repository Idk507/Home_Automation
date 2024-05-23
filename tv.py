import csv
import numpy as np
import tensorflow as tf
from collections import deque
from typing import List, Dict, Tuple, Union

class TV:
    def __init__(self, location: str, power_state: bool, volume: int, channel: int):
        self.location = location
        self.power_state = power_state
        self.volume = volume
        self.channel = channel

    def control_tv(self, power_state: bool, volume: int, channel: int):
        self.power_state = power_state
        self.volume = volume
        self.channel = channel

    def fetch_status(self) -> dict:
        return {
            'location': self.location,
            'power_state': self.power_state,
            'volume': self.volume,
            'channel': self.channel
        }

class Aggregator:
    def __init__(self, time, motion, ac_status, *args, **kwargs):
        self.time = time
        self.motion = motion
        self.ac_status = ac_status
        self.tv_objects = []
        self.ai_agent = None

    def define_tv_objects(self, tv_info: List[Dict[str, Union[str, bool, int]]]):
        for info in tv_info:
            tv_obj = TV(**info)
            self.tv_objects.append(tv_obj)

    def define_ai_object(self, ai_params: dict) -> 'AIAgent':
        self.ai_agent = AIAgent(**ai_params)
        return self.ai_agent

    def invoke_tv_control(self, tv_obj: TV):
        pred_power_state, pred_volume, pred_channel = self.ai_agent.act(self.get_state())
        tv_obj.control_tv(pred_power_state, pred_volume, pred_channel)

    def save_ai_output_to_csv(self, ai_output: str):
        with open('ai_output.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([ai_output])

    def compute_difference(self, previous_state: dict, current_state: dict) -> bool:
        return previous_state != current_state

    def compute_future_difference(self, current_state: dict, next_predicted_state: dict) -> bool:
        return current_state != next_predicted_state

    def send_signal(self):
        print("Signal sent successfully.")

    def get_state(self):
        return np.array([self.time, self.motion, self.ac_status])

class AIAgent:
    def __init__(self, model_type: str, state_dim, action_dim, learning_rate=0.001, gamma=0.95, clip_ratio=0.2):
        self.model_type = model_type
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.agent = PPOAgent(state_dim, action_dim, learning_rate, gamma, clip_ratio)
        self.replay_buffer = deque(maxlen=10000)

    def define_reward_function(self, *args, **kwargs):
        pass

    def train_rl(self, *args, **kwargs):
        states, actions, rewards, next_states, dones = self.sample_from_buffer()
        advantages = self.compute_advantages(rewards, next_states, dones)
        self.agent.train(states, actions, advantages)

    def act(self, state) -> Tuple[bool, int, int]:
        action_prob = self.agent.get_action_prob(state)
        action = np.random.choice(self.action_dim, p=action_prob)
        return True, 50, 10

    def sample_from_buffer(self):
        return None, None, None, None, None

    def compute_advantages(self, rewards, next_states, dones):
        return [0] * len(rewards)

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


tv_info = [
    {'location': 'Living Room', 'power_state': True, 'volume': 50, 'channel': 5},
    {'location': 'Bedroom', 'power_state': False, 'volume': 0, 'channel': 0}
]


aggregator = Aggregator(time=0, motion=True, ac_status=False)
aggregator.define_tv_objects(tv_info)


ai_params = {'model_type': 'Reinforcement Learning', 'state_dim': 3, 'action_dim': 3}
aggregator.define_ai_object(ai_params)


tv_to_control = aggregator.tv_objects[0]
aggregator.invoke_tv_control(tv_to_control)


tv_status = tv_to_control.fetch_status()
print("Current TV status:", tv_status)
