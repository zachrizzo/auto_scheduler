import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Initialize PyGame
pygame.init()
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000
BACKGROUND_COLOR = (30, 30, 30)
ROOM_COLOR = (200, 200, 200)
OBSTACLE_COLOR = (255, 0, 0)
NODE_COLOR = (0, 255, 0)

# Screen setup
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Advanced Classroom Scheduler")
font = pygame.font.Font(None, 24)

# Define constants
schools = ["School A", "School B"]
time_slots = ["9:00", "10:00", "11:00", "14:00", "15:00"]
interests = ["Math", "Science", "History", "Arts"]
locations = ["Classroom", "Recess", "Speech Therapy Room", "Library", "Gym"]

rooms = [
    pygame.Rect(50, 50, 200, 200),
    pygame.Rect(300, 50, 200, 200),
    pygame.Rect(550, 50, 200, 200),
    pygame.Rect(50, 300, 200, 200),
    pygame.Rect(300, 300, 200, 200),
    pygame.Rect(550, 300, 200, 200),
]

student_colors = [
    (255, 0, 0),  # Red
    (0, 255, 0),  # Green
    (0, 0, 255),  # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (255, 165, 0),  # Orange
    (128, 0, 128),  # Purple
]

students = [
    {"name": "Alice", "grade": 9, "school": "School A", "preferred_time": "9:00", "color": student_colors[0], "interest": "Math", "location": "Classroom"},
    {"name": "Bob", "grade": 10, "school": "School A", "preferred_time": "10:00", "color": student_colors[1], "interest": "Science", "location": "Recess"},
    {"name": "Charlie", "grade": 11, "school": "School B", "preferred_time": "11:00", "color": student_colors[2], "interest": "History", "location": "Speech Therapy Room"},
    {"name": "David", "grade": 12, "school": "School B", "preferred_time": "14:00", "color": student_colors[3], "interest": "Arts", "location": "Library"},
    {"name": "Eve", "grade": 9, "school": "School A", "preferred_time": "15:00", "color": student_colors[4], "interest": "Math", "location": "Gym"},
    {"name": "Frank", "grade": 10, "school": "School B", "preferred_time": "9:00", "color": student_colors[5], "interest": "Science", "location": "Classroom"},
    {"name": "Grace", "grade": 11, "school": "School A", "preferred_time": "10:00", "color": student_colors[6], "interest": "History", "location": "Recess"},
    {"name": "Henry", "grade": 12, "school": "School B", "preferred_time": "11:00", "color": student_colors[7], "interest": "Arts", "location": "Speech Therapy Room"},
    {"name": "Ivy", "grade": 9, "school": "School A", "preferred_time": "14:00", "color": student_colors[0], "interest": "Math", "location": "Library"},
    {"name": "Jack", "grade": 10, "school": "School B", "preferred_time": "15:00", "color": student_colors[1], "interest": "Science", "location": "Gym"},
]

teachers = [
    {"name": "Mr. Smith", "school": "School A", "obstacle": "9:00"},
    {"name": "Ms. Johnson", "school": "School B", "obstacle": "14:00"},
]

class ClassroomSchedulerEnv:
    def __init__(self):
        self.students = students
        self.rooms = rooms
        self.reset()

    def step(self, action):
        if self.unscheduled_students:
            student = self.unscheduled_students.pop(0)
            self.schedule[action].append(student)
            reward = self.calculate_reward()
            self.state = self.get_state()
            self.done = len(self.unscheduled_students) == 0
        else:
            reward = 0
            self.done = True
        return self.state, reward, self.done, {}

    def reset(self):
        self.unscheduled_students = self.students.copy()
        self.state = self.get_state()  # This should return a 2D numpy array
        self.done = False
        self.schedule = [[] for _ in range(len(self.rooms))]  # Initialize empty schedule
        return self.state



    def calculate_reward(self):
        reward = 0
        # Calculate the reward based on the grouping, time preference, etc.
        # For simplicity, let's assume some rules here. You can add complex rules.
        for group in self.schedule:
            if len(group) > 4:
                reward -= 10  # Penalty for exceeding group size
            for student in group:
                if student["preferred_time"] in [time_slots[i % len(time_slots)] for i in range(len(self.schedule))]:
                    reward += 1
        return reward

    def get_state(self):
        # Return the state as a normalized vector
        return np.array([
            [student["grade"] / 12, schools.index(student["school"]) / len(schools),
             time_slots.index(student["preferred_time"]) / len(time_slots),
             interests.index(student["interest"]) / len(interests),
             int(any(teacher["obstacle"] == student["preferred_time"] for teacher in teachers)),
             locations.index(student["location"]) / len(locations)]
            for student in self.unscheduled_students
        ])

class PPO(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PPO, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.actor = nn.Linear(64, output_dim)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

    def get_action(self, x):
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, value  # Return the full action tensor

    def evaluate_actions(self, x, action):
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, value, dist_entropy

def train(env, model, optimizer, n_epochs=1000, gamma=0.99, clip_epsilon=0.2, c1=0.5, c2=0.01):
    for epoch in range(n_epochs):
        states = torch.tensor(env.reset(), dtype=torch.float32)
        log_probs = []
        values = []
        rewards = []
        actions = []

        action_tensor, value = model.get_action(states)

        for t, action_item in enumerate(action_tensor):
            next_state, reward, done, _ = env.step(action_item.item())

            log_prob, _, _ = model.evaluate_actions(states[t].unsqueeze(0), action_item.unsqueeze(0))

            log_probs.append(log_prob)
            values.append(value[t])
            rewards.append(reward)
            actions.append(action_item.item())

            if done:
                break

        # Convert lists to tensors
        log_probs = torch.cat(log_probs)
        values = torch.cat(values)
        rewards = torch.tensor(rewards)
        actions = torch.tensor(actions)

        # Calculate returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)

        advantages = returns - values

        # PPO Update
        for _ in range(10):
            # Recompute log probs, values, and entropy for each update
            new_log_probs, new_values, entropy = model.evaluate_actions(states, actions)
            ratio = torch.exp(new_log_probs - log_probs.detach())
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages

            loss = -torch.min(surr1, surr2).mean() + c1 * (returns - new_values.squeeze()).pow(2).mean() - c2 * entropy.mean()

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        print(f'Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}')

# Initialize environment and model
env = ClassroomSchedulerEnv()
model = PPO(input_dim=6, output_dim=len(rooms))
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Train model
train(env, model, optimizer)

pygame.quit()
