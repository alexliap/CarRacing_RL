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
# env = RecordVideo(env, 'car_racing')

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

a_size = env.action_space.n
# somewhere I saw that only one nn was used, so I can try to use only q_function
q_function = Network([1, 16, 32, 64], [9216, 256, a_size], 0).to(device)
# mirror = Network([1, 16, 32, 64], [9216, 256, a_size], 0).to(device)

optim = SGD(q_function.parameters(), lr = 2.5e-4)

buffer = []
total_rewards = []

max_episodes = 1000
max_steps = 1000
max_epsilon = 1
min_epsilon = 0.05
decay_rate = 0.001
gamma = 0.98
batch_size = 128

for episode in tqdm(range(max_episodes), position = 0, desc = 'Episode progress'):
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    # to torch and grayscale
    state = tensor_to_grayscale(env.reset()[0]).to(device)
    step = 0
    done = False
    total_reward = 0
    for i in tqdm(range(max_steps), position = 1, desc = 'Episode step progress', leave = False,
                  mininterval = 60):
        action = epsilon_greedy_policy(env, q_function, state, epsilon)

        new_state, reward, done, truncated, info = env.step(action)
        # to torch and grayscale
        new_state = tensor_to_grayscale(new_state).to(device)
        total_reward += reward

        buffer.append((state, action, reward, new_state))
        # begin training when buffer has over 128 experiences
        if len(buffer) > batch_size:
            train_batch = random.sample(buffer, batch_size)
            # --------------------- no problem --------------------- #
            # train on past experiences
            losses = []
            for pair in train_batch:
                loss = q_loss(q_function, pair[0], pair[1], pair[3], pair[2], gamma)
                losses.append(loss)

            mean_loss = torch.mean(torch.stack(losses))

            if done:
                break

            optim.zero_grad()
            mean_loss.backward()
            optim.step()

        state = new_state
        # mirror.parameters = q_function.parameters()

        # if i % 250 == 0:
        #     print(f'\n Step {i}: {total_reward}')

    # if episode % 5 == 0:
    #     print(f'Episode {episode}: {total_reward}')
    print(f'\n Episode {episode}: {total_reward}')
    total_rewards.append(total_reward)
