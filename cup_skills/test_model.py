from sklearn.gaussian_process import GaussianProcessRegressor
from random import random
from sklearn.gaussian_process import GaussianProcess
import numpy as np
import pdb
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 30
from keras.models import Sequential
from keras.layers import Dense, Activation, GaussianDropout, Dropout
from keras.optimizers import Adam
from keras import regularizers

from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, RationalQuadratic, ExpSineSquared, Matern

#kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
kernel = C(5.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
#kernel = C(25.0, (1e-3, 1e3)) * RBF(1, (1e-1, 1e1))+RationalQuadratic(alpha=0.1, length_scale=0.957)
#kernel = Matern(nu=2.5, length_scale = 0.1)

#kernel = C(50, (1e-3, 10e1)) * RBF(1)

gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=10)
#gp = GaussianProcess(theta0=0.1, nugget = 0.1) this works pretty well

nn = Sequential()
nn.add(Dense(2, input_dim=2, activation="linear"))
nn.add(Dense(1, activation="linear"))
opt = Adam(lr=0.01)
nn.compile(loss="mse", optimizer=opt)



samples = np.load("dataset/samples_1_two_params.npy")
rewards = np.load("dataset/rewards_1_two_params.npy")
samples = np.load("dataset/samples_1_two_spread_params.npy")
rewards = np.load("dataset/rewards_1_two_spread_params.npy")

samples_test = np.load("dataset/samples_1_two_params_test.npy")
rewards_test = np.load("dataset/rewards_1_two_params_test.npy")

gp.fit(samples, rewards)
#nn.fit(samples, rewards, epochs =3, batch_size = 20)
#print("Log likelihood", gp.log_marginal_likelihood())

def plot_samples_and_rewards(samples, rewards):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    xs = samples[:,0]
    ys = samples[:,1]
    zs = rewards
    cs = []
    for i in range(samples.shape[0]):
        if rewards[i] > 39:
            color = 'g'
        else: 
            color = 'r'
        cs.append(color)
    #ax.scatter(xs, ys, zs, c=c, marker=m)
    ax.scatter(xs, ys, zs, c =cs, marker = 'o', s=85)

    ax.set_xlabel('offset')
    ax.set_ylabel('velocity')
    ax.set_zlabel('reward')
    plt.show()

def plot_samples_and_rewards_1D(samples, rewards, predictions):
    offset_samples = samples[:,0]
    prediction_points = np.array([pt[0] for pt in predictions])
    stds = np.array([pt[1] for pt in predictions])
    plt.scatter(offset_samples, rewards, color="r", label="data")
    plt.scatter(offset_samples, prediction_points, color="b", marker = '^', label="prediction")
    plt.xlabel("Offset value")
    plt.ylabel("Reward")
    plt.legend()
    plt.show()

def add_noise(samples, A):
    return np.random.normal(samples, A)

def test(samples, use_nn=False):
    total_error = []
    predictions = []
    for i in range(samples.shape[0]):
        if not use_nn:
  
	    _, stdev = gp.predict([samples[i]], return_std=True)
	    
	    predicted = _.item()
            predictions.append((predicted, stdev.item()))
        
        else:
            predicted = nn.predict(np.matrix(samples[i]))
        actual = rewards[i].item()
        error = (predicted-actual)**2
        total_error.append(error)
    print(np.median(total_error)**0.5, "median error")
    print(np.mean(total_error)**0.5, "mean error")
    return np.median(total_error)**0.5, predictions


def test_set(use_nn=False):
    print("Error", test(samples_test, use_nn=use_nn)[0])

def test_noise(use_nn=False):
    noise_levels = [0.8, 0.3, 0.15, 0.09, 0.05,0.01, 0.001, 0.0001, 0]
    noise_levels.reverse()

    for noise_level in noise_levels:
         errors = []
         for i in range(15):
	     noisy_samples = add_noise(samples, noise_level)
             errors.append(test(noisy_samples, use_nn=use_nn)[0])
         print("noise_level", noise_level, "average noise", sum(errors)/len(errors))


def uniform_random_sample():
    lower = [-0.2, 0.9]
    upper = [0.2, 2.5]
    sample = []
    for i in range(len(lower)):
        sample.append((upper[i] - lower[i]) * random()+ lower[i])
    return sample



def optimize(model):
    #picks the value that's highest in the GP
    N = 5
    scores = []
    for i in range(N):
        random_action = uniform_random_sample()
        score, stdev = model.predict([random_action], return_std = True)
        scores.append(score+stdev)
    best_score_i = np.argmax(scores)
    best_score = scores[best_score_i]

#test_set(use_nn=False)
#test_noise(use_nn=False)
#noisy_samples = add_noise(samples, 0.01)
predictions = test(samples_test)[1]
plot_samples_and_rewards_1D(samples_test, rewards_test, predictions)
#optimize(gp)
       

