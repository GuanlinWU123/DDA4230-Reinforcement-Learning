import gym
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.distributions import Categorical

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ACTION_SPACE = [0, 1]
RENDER = False

# you can fine-tune these parameters to achieve better results
EPISODES = 2000
STEPS = 1000
GAMMA = 0.9
learning_rate = 3e-4
hidden_size = 128
env = gym.make("InvertedPendulum-v4")
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]

# calculate the return
# we provide the code
def compute_returns(next_value, rewards, masks, gamma=GAMMA):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


# define the model
class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        # initialization
        # TODO
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mean_layer = nn.Linear(hidden_size, action_size)
        self.log_std = nn.Parameter(torch.zeros(1, action_size))

    def forward(self, state):
        # remember the definition of actor critic algorithm
        # you need to output the action distribution here
        # hint: use actor to predict mean and std (or log_std) to form the Gaussian distribution
        # Note: For the sampled action, we should use action.detach() to avoid calculate its gradient,
        # because the gradient is for log_prob
        # TODO
        if isinstance(state, list):
            state = np.array(state, dtype=float)
        state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        x = F.relu(self.fc1(state))
        x = F.tanh(self.fc2(x))
        mean = 3 * torch.tanh(self.mean_layer(x))
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)
        distribution = torch.distributions.Normal(mean, std)
        # distribution = torch.distributions.Categorical(F.softmax(x, dim=-1))
        return distribution


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        # initialization
        # TODO
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.value_head = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        # remember the definition of actor critic algorithm
        # for critic, we need to output the predicted reward value
        # hint: output a value with dim 1.
        # TODO
        state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        x = F.relu(self.fc1(state))
        x = F.tanh(self.fc2(x))
        value = self.value_head(x)
        return value


# training the model
actor = Actor(state_size, action_size).to(DEVICE)
critic = Critic(state_size, action_size).to(DEVICE)

optimizerA = torch.optim.Adam(actor.parameters(), lr=learning_rate)
optimizerC = torch.optim.Adam(critic.parameters(), lr=learning_rate)

all_rewards = []
for episode in range(EPISODES):
    done = False

    state = env.reset()
    state = state[0]
    # record log_probability, predicted_value, reward, mask = 1-done, and entropy
    log_probs = []
    values = []
    rewards = []
    masks = []
    entropy = 0

    for step in range(STEPS):
        # if RENDER:
        #     env.render()
        state = torch.FloatTensor(state).to(DEVICE)

        # you should perform the game and record the infos for training
        # hint: at current state
        # use the actor to predict the distribution, then get action with 'distribution.sample()'
        # use the critic to predict the reward value
        # then use env.step(action) to get the next state (you may meet the error about tensor/numpy, use action.cpu().numpy())
        # remember to record log_prob and entropy (we provided)
        # do above until done
        # TODO

        distribution = actor(state)
        value = critic(state)
        action = distribution.sample()
        scaled_action = action * 3
        action = scaled_action.clamp(env.action_space.low[0], env.action_space.high[0])  # Clamp to action space limits
        # we provide how to record log_prob and entropy
        log_prob = distribution.log_prob(action)
        entropy += distribution.entropy().mean()

        # use list.append to record
        # TODO
        next_state, reward, done, _, _ = env.step(action.cpu().detach().numpy()[0])
        rewards.append(torch.tensor([reward], dtype=torch.float32, device=DEVICE))
        masks.append(torch.tensor([1 - done], dtype=torch.float32, device=DEVICE))
        values.append(value)
        log_probs.append(log_prob)
        state = next_state

        if done:
            sum = 0
            for reward in rewards:
                sum = sum + reward.cpu().numpy()
            all_rewards.append(sum)
            if episode%100 ==0:
                print(f"EPISODE {episode} SCORE: {sum} roll{pd.Series(all_rewards).tail(30).mean()}")
            break

    # for updating actor and critic
    # here we use GAE to compute the return
    next_state = torch.FloatTensor(next_state).to(DEVICE)
    next_value = critic(next_state)
    returns = compute_returns(next_value, rewards, masks, GAMMA)

    log_probs = torch.cat(log_probs)
    returns = torch.cat(returns).detach()
    values = torch.cat(values)

    # calculate advantage
    advantage = returns - values

    # calculate actor loss and critic loss here
    # hint: actor loss is related to log_probs and advantage, critic is trying to make advantage=0 (i.e., td error between return and value)
    # or if you feel hard to implement actor-critic with advantage,
    # you can ignore advantage and implement the original actor-critic as you like
    # TODO
    actor_loss = -(log_probs * advantage.detach()).mean()
    critic_loss = advantage.pow(2).mean()

    optimizerA.zero_grad()
    optimizerC.zero_grad()
    actor_loss.backward()
    critic_loss.backward()
    optimizerA.step()
    optimizerC.step()


# your task: evaluate the performance
# TODO
plt.figure(figsize=(10, 6))
plt.plot(all_rewards, label='Episodic Return H_128 G_0.95 Lr_3e-4')
plt.xlabel('Episode')
plt.ylabel('Discounted Cumulative Reward')
plt.title('Training Progress of Actor-Critic Algorithm')
plt.legend()
plt.grid(True)
plt.show()

# for visulizing the results, provided

# import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import cv2  # if you miss this package, you should run "pip install opencv-python"

actor.eval()
font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 500)
fontScale = 1
fontColor = (255, 255, 255)
lineType = 2

fig = plt.figure()
# ACTION_SPACE = [0, 1]
env = gym.make("InvertedPendulum-v4", render_mode='rgb_array')
env = env.unwrapped
state = env.reset()
state = state[0]
ims = []
rewards = []
for step in range(500):
    # env.render()
    img = env.render()
    # print(img)
    state = torch.FloatTensor(state).to(DEVICE)
    action = actor(state).sample()
    scaled_action = action * 3
    action = scaled_action.clamp(env.action_space.low[0], env.action_space.high[0])
    action = action.cpu().detach().numpy()[0]
    print("action: ", action)
    state, reward, done, _, _ = env.step(action)
    print("state: ", state, "type: ", type(state))
    print("reward: ", reward)
    rewards.append(reward)
    print(img.shape)
    cv2_im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im_rgb)

    draw = ImageDraw.Draw(pil_im)

    # Draw the text
    draw.text((0, 0), f"Step: {step} Action : {action} Reward: {reward} Total Rewards: {np.sum(rewards)} done: {done}",
              fill="#000000")

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
im_ani.save('/Users/guanlinwu/Desktop/DDA4230/Project_4/invertedPendulum_AC.gif', writer=writer)

