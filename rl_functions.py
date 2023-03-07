import torch
import numpy as np
from torchvision.transforms.functional import rgb_to_grayscale
import torch.nn.functional as F


def epsilon_greedy_policy(env, q_function, state, epsilon):
    # Randomly generate a number between 0 and 1
    random_num = np.random.uniform(0, 1)
    # if random_num > greater than epsilon --> exploitation
    if random_num > epsilon:
        # Take the action with the highest value given a state
        # np.argmax can be useful here
        action = torch.argmax(q_function.forward(state)).item()
    # else --> exploration
    else:
        action = env.action_space.sample()

    return action


def q_loss(q_function, state, action, new_state, reward, gamma):
    with torch.no_grad():
        target = reward + gamma * q_function.forward(new_state).max()
    expected = q_function.forward(state)[action]
    return F.l1_loss(expected, target)


def tensor_to_grayscale(img):
    img = torch.from_numpy(img.astype('float32')).view(3, 96, -1)
    img = rgb_to_grayscale(img, num_output_channels = 1)
    return img
