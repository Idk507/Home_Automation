from typing import List, Dict, Union, Any, Tuple
import random
import datetime
import time
import json
import csv
from collections import deque
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import secrets
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

#class Fan
class Fan:
    def __init__(self, fan_speed: int, location: str, *args, **kwargs):
        self.fan_speed = fan_speed 
        self.location = location
    def control_fan(self, speed: int):
        #  to control fan speed
        self.Fan_Speed = speed
    def fetch_status(self) -> dict:
        # to fetch current status
        return {"Fan_Speed": self.Fan_Speed, "Location": self.location}
#Aggregator class

class Aggregator:
    def __init__(self, temperature, time, motion, AC_temp_status, *args, **kwargs):
        self.temperature = temperature
        self.time = time
        self.motion = motion
        self.AC_temp_status = AC_temp_status
        self.fan_objects = []
        self.ai_agent = None
        self.ai_output_log = []

    def define_fan_objects(self, fan_info: List[Dict[str, Union[int, str]]]):
        for info in fan_info:
            fan_obj = Fan(Fan_Speed=info["Fan_Speed"], location=info["Location"])
            self.fan_objects.append(fan_obj)

    def define_ai_object(self, ai_params: dict) -> AIAgent:
        self.ai_agent = AIAgent(model_type=ai_params.get('model_type', 'deep_learning'))
        return self.ai_agent

    def invoke_fan_control(self, fan_obj: Fan):
        # fan control based on AI predictions
        if self.ai_agent:
            state, confidence = self.ai_agent.predict({"temperature": self.temperature, "motion": self.motion})
            if state == 1 and confidence >= 0.85:
                fan_obj.control_fan(5)  # Set fan speed to 5 if predicted state is ON with confidence >= 85%

    def save_ai_output_to_csv(self, ai_output: str):
        # Save output to CSV for the last 15 days
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.ai_output_log.append({"timestamp": timestamp, "output": ai_output})

        # Save the last 15 days' data
        last_15_days_data = [log for log in self.ai_output_log if datetime.now() - datetime.strptime(log["timestamp"], "%Y-%m-%d %H:%M:%S") <= timedelta(days=15)]

        with open('ai_output_log.csv', 'w', newline='') as csvfile:
            fieldnames = ["timestamp", "output"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(last_15_days_data)

    def compute_difference(self, previous_state: dict, current_state: dict) -> bool:
        #  difference between previous and current states
        difference = any(previous_state.get(key) != current_state.get(key) for key in current_state)
        manual_override = difference
        return manual_override

    def compute_future_difference(self, current_state: dict, next_predicted_state: dict) -> bool:
        #  difference between current and predicted states
        difference = any(current_state.get(key) != next_predicted_state.get(key) for key in current_state)
        return difference

    def send_signal(self):
        #  sending a signal
        print("Sending signal")
    
    
    def fetch_status(self, fan_objects: List[Fan]):
        statuses = []
        for fan in fan_objects:
            statuses.append(fan.fetch_status())
        return statuses



class AIAgent:
    def __init__(self, model_type: str, *args, **kwargs):
        self.model_type = model_type

        if model_type == 'ReinforcementLearning':
            # Initialize PPO model
            self.policy = self.build_policy_network()
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def build_policy_network(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(4,)),  # Assuming 4 features in the state
            tf.keras.layers.Dense(2, activation='softmax')  # Assuming binary action space (ON or OFF)
        ])
        return model

    def train_rl(self, dataset: List[Dict[str, Any]]):
        # Train the PPO model
        states = np.array([list(sample['input'].values()) for sample in dataset])
        actions = np.array([sample['label']['fan_state'] for sample in dataset])
        advantages = np.array([self.define_advantage(sample['label']['confidence']) for sample in dataset])

        actions_categorical = tf.keras.utils.to_categorical(actions, num_classes=2)

        with tf.GradientTape() as tape:
            policy_prob = self.policy(states, training=True)
            chosen_prob = tf.reduce_sum(actions_categorical * policy_prob, axis=1)
            old_policy_prob = tf.stop_gradient(chosen_prob)

            ratio = tf.exp(tf.math.log(chosen_prob + 1e-10) - tf.math.log(old_policy_prob + 1e-10))
            clipped_ratio = tf.clip_by_value(ratio, 1 - 0.2, 1 + 0.2)
            surrogate = -tf.minimum(ratio * advantages, clipped_ratio * advantages)
            loss = tf.reduce_mean(surrogate)

        gradients = tape.gradient(loss, self.policy.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))

    def act(self, input_data: Dict[str, Any]) -> str:
        # Take an action based on the current state using the trained policy network
        state = np.array(list(input_data.values())).reshape(1, -1)
        policy_prob = self.policy(state, training=False).numpy()
        action = np.random.choice([0, 1], p=policy_prob.flatten())

        return 'ON' if action == 1 else 'OFF'

    def define_advantage(self, confidence: float) -> float:
       
        return confidence
    
    
ppo_agent = AIAgent(model_type='ReinforcementLearning')

# Train the PPO model
ppo_agent.train_rl(dummy_rl_dataset)

current_state = {'temperature': 25.0, 'time': 15, 'motion': True, 'AC_temp_status': 22.0}
action = ppo_agent.act(current_state)
print(f"Recommended action based on PPO: {action}")   
    

"""class AIAgent:
    def __init__(self, model_type: str, *args, **kwargs):
        self.model_type = model_type

        if model_type == 'ReinforcementLearning':
            # Initialize PPO model
            self.policy = self.build_policy_network()
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def build_policy_network(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),  # Assuming 11 features in the state
            tf.keras.layers.Dense(2, activation='softmax')  # Assuming binary action space (ON or OFF)
        ])
        return model
  

    def act(self, state: Dict[str, Any]) -> int:
        # Convert state to numpy array
        state_array = np.array(list(state.values()))
        state_array = np.expand_dims(state_array, axis=0)

        # Predict action probabilities
        policy_prob = self.policy(state_array, training=False)

        #  the action with the highest probability
        action = np.argmax(policy_prob[0])

        return 'ON' if action == 1 else 'OFF'

    def train_rl(self, dataset: List[Dict[str, Any]]):
    # Train the PPO model
        states = np.array([list(sample['input'].values()) for sample in dataset])
        print(states.shape)
        actions = np.array([sample['TargetFanState'] for sample in dataset])
        advantages = np.array([self.define_advantage(sample['confidence']) for sample in dataset])

        actions_categorical = tf.keras.utils.to_categorical(actions, num_classes=2)

        with tf.GradientTape() as tape:
            policy_prob = self.policy(states, training=True)
            chosen_prob = tf.reduce_sum(actions_categorical * policy_prob, axis=1)
            old_policy_prob = tf.stop_gradient(chosen_prob)

            ratio = tf.exp(tf.math.log(chosen_prob + 1e-10) - tf.math.log(old_policy_prob + 1e-10))
            clipped_ratio = tf.clip_by_value(ratio, 1 - 0.2, 1 + 0.2)
            surrogate = -tf.minimum(ratio * advantages, clipped_ratio * advantages)
            loss = tf.reduce_mean(surrogate)

        gradients = tape.gradient(loss, self.policy.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))

    def define_advantage(self, confidence: float) -> float:
        return confidence

dummy_rl_dataset = [
    {
        'input': {
            'Temperature': 25.0,
            'TimeOfDay': 15.0,
            'Date': 5,
            'Month': 1,
            'DayOfWeek': 3,
            'MotionSensorState': 1,
            'ACTemperatureStatus': 0,
            'CurrentFanSpeed': 7,
            'Location': 2,
            'PreviousFanState': 1,
        },
        'TargetFanState': 1,  # Assuming the target fan state is ON
        'confidence': 0.9  
    }
]


ppo_agent = AIAgent(model_type='ReinforcementLearning')
ppo_agent.train_rl(dummy_rl_dataset)


current_state = {
    'Temperature': 25.0,
    'TimeOfDay': 15.0,
    'Date': 5,
    'Month': 1,
    'DayOfWeek': 3,
    'MotionSensorState': 1,
    'ACTemperatureStatus': 0,
    'CurrentFanSpeed': 7,
    'Location': 2,
    'PreviousFanState': 1,
}

action = ppo_agent.act(current_state)
print(f"Recommended action based on PPO: {action}")

        
      
        





import random
from datetime import datetime, timedelta

# Function to generate dummy dataset for Fan class Deep Learning model
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Function to create a dummy dataset
def create_dummy_dataset(num_samples=1000):
    np.random.seed(42)

    # Generate random data for the environment variables
    temperature = np.random.uniform(18.0, 30.0, size=num_samples)
    time_of_day = np.random.uniform(0.0, 24.0, size=num_samples)
    date = np.random.randint(1, 31, size=num_samples) # Assuming 31 days in a month 
    month = np.random.choice(12, size=num_samples)
    day_of_week = np.random.choice([2,3,4,5,6,7,1],#monday -2,tuesday-3,wednesday-4,thursday-5,friday-6,saturday-7,sunday-1
                                   size=num_samples)
    motion_sensor_state = np.random.choice([1,0], size=num_samples)#1-True,0-False
    ac_temperature_status = np.random.choice([1,0], size=num_samples) 

    # Generate random data for the fan variables
    current_fan_speed = np.random.randint(0, 11, size=num_samples)
    location = np.random.choice([1,2,3], size=num_samples) #1-Living Room,2-Bedroom,3-Kitchen
    previous_fan_state = np.random.choice([1,0,-1], size=num_samples)

    # Generate random data for the target variable
    target_fan_state = np.random.choice([1,0], size=num_samples)

    # Create a DataFrame
    df = pd.DataFrame({
        'Temperature': temperature,
        'TimeOfDay': time_of_day,
        'Date': date,
        'Month': month,
        'DayOfWeek': day_of_week,
        'MotionSensorState': motion_sensor_state,
        'ACTemperatureStatus': ac_temperature_status,
        'CurrentFanSpeed': current_fan_speed,
        'Location': location,
        'PreviousFanState': previous_fan_state,
        'TargetFanState': target_fan_state
    })

    return df

# Create a dummy dataset
dummy_dataset = create_dummy_dataset()

# Display the first few rows of the dataset
print(dummy_dataset.head())


# Create a dummy dataset
dummy_dataset = create_dummy_dataset()


df = dummy_dataset
X = df.drop('TargetFanState', axis=1)
y = df['TargetFanState']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build a deep learning model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

"""