import gym
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RENDER=False
env = gym.make("InvertedPendulum-v4")

num_input = env.observation_space.shape[0] # the cartpole env has 4 observations
num_output = env.action_space.shape[0] # the cartpole env has 2 actions, 0 and 1
# ACTION_SPACE = [0,1]

# you can fine-tune these parameters to achieve better results
EPISODES = 2000
STEPS = 500
GAMMA=0.9
learning_rate = 3e-4
hidden_size = 128
# print("input:", num_input, "output: ", num_output)
# define the model
class ReinforceModel(nn.Module):
    def __init__(self, num_input, num_output):
        super(ReinforceModel, self).__init__()
        self.fc1 = nn.Linear(num_input, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 32)
        self.mean_layer = nn.Linear(32, num_output)
        # self.sigma_head = nn.Linear(32, num_output)
        self.log_std = nn.Parameter(torch.zeros(1, num_output))

    def forward(self, x):
        if isinstance(x, list):
            x = np.array(x, dtype=float)
        # print("1: ", x)
        # elif isinstance(x, np.ndarray):
        #     pass  # x is already a numpy array
        # else:
        #     raise TypeError("Input x must be a numpy array or a list of numpy arrays")
        x = torch.tensor(x, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        # print("2: ", x)
        x = F.relu(self.fc1(x))
        x = F.tanh(self.fc2(x))
        # mu = self.mu_head(x)
        # sigma = torch.clamp(self.sigma_head(x), min=1e-5, max=1)
        mean = 3 * torch.tanh(self.mean_layer(x))
        log_std = self.log_std.expand_as(mean)
        mu = mean
        sigma = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, sigma)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob


# training the model
# print(num_input, num_output)
model = ReinforceModel(num_input, num_output).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
all_rewards =[] # record cumulative rewards
for episode in range(EPISODES):
    state = env.reset()
    log_probs = []
    rewards = []
    # print("state: ", state, "type: ",  type(state))
    state = state[0]
    for step in range(STEPS):
        action, log_prob = model(state)
        # print("action before clamp: ", action)
        scaled_action = action * 3
        action = scaled_action.clamp(env.action_space.low[0], env.action_space.high[0])  # Clamp to action space limits
        # print("action after clamp: ", action)
        state, reward, done, _, _ = env.step(action.cpu().detach().numpy()[0])
        log_probs.append(log_prob)
        rewards.append(reward)
        if done:
            break

    all_rewards.append(np.sum(rewards))
    if episode % 100 == 0:
        print(f"EPISODE {episode} SCORE: {np.sum(rewards)}")

    discounted_rewards = []
    for t in range(len(rewards)):
        Gt = sum([GAMMA ** i * rewards[i+t] for i in range(len(rewards)-t)])
        discounted_rewards.append(Gt)

    discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32, device=DEVICE)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-7)

    policy_gradient = []
    for log_prob, Gt in zip(log_probs, discounted_rewards):
        policy_gradient.append(-log_prob * Gt)

    optimizer.zero_grad()
    policy_gradient = torch.cat(policy_gradient).sum()
    policy_gradient.backward()
    optimizer.step()

# your task: evaluate the performance
# TODO
plt.figure(figsize=(10, 6))
plt.plot(all_rewards, label='Episodic Return H_128 G_0.9 Lr_3e-4')
plt.xlabel('Episode')
plt.ylabel('Discounted Cumulative Reward')
plt.title('Training Progress of REINFORCE Algorithm')
plt.legend()
plt.grid(True)
plt.show()


# for visulizing the results, provided, do not need to change

import matplotlib.animation as animation
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import cv2 # if you miss this package, you should run "pip install opencv-python"
model.eval()
font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale = 1
fontColor = (255,255,255)
lineType = 2

fig = plt.figure()
env = gym.make("InvertedPendulum-v4", render_mode='rgb_array')
env = env.unwrapped
state = env.reset()
# print("state: ", state, "type: ",  type(state))
state = state[0]

ims = []
rewards = []
for step in range(500):
    # env.render()
    img = env.render()
    # print(img)
    action,log_prob = model(state)
    scaled_action = action * 3
    action = scaled_action.clamp(env.action_space.low[0], env.action_space.high[0])
    print("action: ", action)
    state, reward, done, _, _ = env.step(action.cpu().detach().numpy()[0])
    print("state: ", state, "type: ", type(state))
    print("reward: ", reward)
    rewards.append(reward)
    print(img.shape)
    cv2_im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im_rgb)

    draw = ImageDraw.Draw(pil_im)

    # Draw the text
    draw.text((0, 0), f"Step: {step} Action : {action} Reward: {reward} Total Rewards: {np.sum(rewards)} done: {done}",fill="#000000")

    # Save the image
    img = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
    # img = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2GRAY)
    im = plt.imshow(img, animated=True)
    ims.append([im])
env.close()

Writer = animation.writers['pillow']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
im_ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=3000,
                                    blit=True)
im_ani.save('/Users/guanlinwu/Desktop/DDA4230/Project_4/invertedPendulum_R.gif', writer=writer)