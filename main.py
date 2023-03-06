import gym
from model import Network
from gym.wrappers import RecordVideo
from rl_functions import *
from torch.optim import SGD
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

env = gym.make('CarRacing-v2', render_mode = 'rgb_array',
               lap_complete_percent = 100, continuous=False)
env = RecordVideo(env, 'car_racing')

device = torch.device("mps")

a_size = env.action_space.n

q_function = Network([3, 16, 32, 64], [9216, 256, 5], 0.2).to(device)

mirror = Network([3, 16, 32, 64], [9216, 256, 5], 0.2).to(device)

optim = SGD(q_function.parameters(), lr = 1e-4)

buffer = []
total_rewards = []

max_episodes = 10000
max_steps = 1000
epsilon = 1
decay_rate = 0.0005
gamma = 0.98
for episode in tqdm(range(max_episodes), position = 0, desc = 'Episode progress'):
    state = torch.from_numpy(env.reset()[0].astype('float32')).view(3, 96, -1).to(device)
    step = 0
    done = False
    total_reward = 0
    for i in tqdm(range(max_steps), position = 1, desc = 'Episode step progress', leave = True,
                  mininterval = 60):
        action = epsilon_greedy_policy(env, q_function, state, epsilon)

        new_state, reward, done, truncated, info = env.step(action)
        new_state = torch.from_numpy(new_state.astype('float32')).view(3, 96, -1).to(device)
        total_reward += reward

        buffer.append((state, action, reward, new_state))
        train_batch = random.choices(buffer, k = 32)
        # --------------------- no problem --------------------- #
        # train on past experiences
        losses = []
        for pair in train_batch:
            loss = q_loss(q_function, mirror, pair[0], pair[1], pair[3], pair[2], gamma)
            losses.append(loss)

        mean_loss = torch.mean(torch.stack(losses))

        if done:
            break

        optim.zero_grad()
        mean_loss.backward()
        optim.step()

        state = new_state
        mirror.parameters = q_function.parameters()
        epsilon = epsilon * np.exp(-decay_rate * episode)

        if i % 250 == 0:
            print(f'\n Step {i}: {total_reward}')

    if episode % 5 == 0:
        print(f'Episode {episode}: {total_reward}')
    total_rewards.append(total_reward)
