from sklearn.gaussian_process import GaussianProcessRegressor
from random import random
from matplotlib import cm
from sklearn.gaussian_process import GaussianProcess
import numpy as np
import pdb
from mpl_toolkits.mplot3d import Axes3D
from pdg_learn import load_datasets
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 18
from keras.models import Sequential
from keras.layers import Dense, Activation, GaussianDropout, Dropout
from keras.optimizers import Adam
from keras import regularizers

from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, RationalQuadratic, ExpSineSquared, Matern, WhiteKernel


#gp = GaussianProcess(theta0=0.1, nugget = 0.1) this works pretty well

def num_params_v_fit():
    lls = []
    fits = []
    num_params = 6
    samples = np.load("dataset/samples_1_biger.npy")
    rewards = np.load("dataset/rewards_1_biger.npy")
    x = np.linspace(1,num_params, num_params)
    for i in range(1,num_params+1):
        fit, ll = fit_and_test_n_dim(samples, rewards, i)
        lls.append(ll)
        fits.append(fit)
    plot_fit_and_ll(x, fits, lls)

def plot_fit_and_ll(x, fits, lls):
    plt.title("Log marginal likelihood over number of parameters")
    plt.xlabel("Number of parameters")
    plt.ylabel("Log likelihood")
    plt.scatter(x, lls)
    plt.show()
    
    plt.title("Mean squared error over number of parameters")
    plt.xlabel("Number of parameters")
    plt.ylabel("MSE")
    plt.scatter(x,fits)
    plt.show()
        
        
    
def fit_and_test_n_dim(samples, rewards, n, alpha=0.1, length_scale=0.1, use_nn = False):
    if use_nn:
	nn = Sequential()
	nn.add(Dense(32, input_shape=(8,), activation="relu"))
	nn.add(Dense(64,  activation="relu"))
	nn.add(Dense(1, activation="sigmoid"))
	opt = Adam(lr=0.01)
	nn.compile(loss="mse", optimizer=opt)
    else:
	gp_kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale, (1e-2, 1e2))
	gp = GaussianProcessRegressor(kernel=gp_kernel, n_restarts_optimizer=8, alpha=alpha)
    num_training = 1800
    relevant_samples_training = samples[:num_training,:n ]
    relevant_rewards_training = rewards[:num_training]
    relevant_samples_test = samples[num_training:,:n ]
    relevant_rewards_test = rewards[num_training:]
    if use_nn:
        nn.fit(relevant_samples_training, relevant_rewards_training, epochs=3800, batch_size=50)
        ll = 0 
        predictions = nn.predict_on_batch(relevant_samples_test)
    else:
        print("before fitting")
	gp.fit(relevant_samples_training, relevant_rewards_training)
        print("after fitting")
	ll = gp.log_marginal_likelihood()
	predictions = gp.predict(relevant_samples_test)    
    
    se = (predictions-relevant_rewards_test)**2
    
    mse = np.mean(se)
    return mse, ll


#Needs to be vel, total, offset, or height
def fit_and_test_one_dim(var):
    samples_file = np.load("dataset/samples_vary_"+var+".npy")
    rewards_file = np.load("dataset/rewards_vary_"+var+".npy")
    names = ["offset", "height", "vel", "total"]
    col_index = names.index(var)
    gp_kernel = C(1.0, (1e-3, 1e3)) * RBF(0.1, (1e-2, 1e2))+WhiteKernel(5.0)
    gp_1D = GaussianProcessRegressor(kernel=gp_kernel, n_restarts_optimizer=8, alpha=0.1)
    num_training = 75
    training_samples = samples_file[:num_training,col_index]
    training_rewards = rewards_file[:num_training]
    test_samples = samples_file[num_training:,col_index]
    test_rewards = rewards_file[num_training:]
    gp_1D.fit(training_samples.reshape(-1, 1) , training_rewards)
    predictions = gp_1D.predict(test_samples.reshape(-1, 1) , return_std=True)
    plot_samples_and_rewards_1D(test_samples, test_rewards, predictions, gp=gp_1D, column_index = col_index, label=var)

#var in form offset_height, offset_vel, offset_total
def fit_and_test_two_dim(var, samples_file = None, rewards_file = None):
    if samples_file is None:
        samples_file = np.load("dataset/samples_vary_"+var+".npy")
        rewards_file = np.load("dataset/rewards_vary_"+var+".npy")
    dims = [0,1]
    num_training = 19
    assert(len(dims) == 2)
    training_samples = np.hstack([np.matrix(samples_file[:num_training,dims[0]]).T, 
                                  np.matrix(samples_file[:num_training,dims[1]]).T])
    training_rewards = rewards_file[:num_training]
    test_samples = np.hstack([np.matrix(samples_file[num_training:,dims[0]]).T, 
                                  np.matrix(samples_file[num_training:,dims[1]]).T])
    test_rewards = rewards_file[num_training:]
    
    gp_kernel = C(0.01, (1e-3, 1e3)) * RBF(0.6, (1e-2, 1e2))
    gp_2D = GaussianProcessRegressor(kernel=gp_kernel, n_restarts_optimizer=8, alpha=1e-3)
    gp_2D.fit(training_samples, training_rewards)
    predictions = gp_2D.predict(test_samples, return_std=True)
    se = (predictions[0].reshape(test_rewards.shape)-test_rewards)**2
    mse = np.mean(se)
    print(np.round(predictions[0],2))
    print("test samples", test_rewards)
    pdb.set_trace()
    print(mse, "mean squared error")
    plot_samples_and_rewards_2D(test_samples, test_rewards, predictions, gp=gp_2D, label=var)


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

def plot_samples_and_rewards_2D(samples, rewards, predictions, gp=None, label=""):
    prediction_points = np.array([pt[0] for pt in predictions[0]])
    stds = np.array([pt for pt in predictions[1]])
    fig = plt.figure()
    dims = label.split('_')
    
    ax = fig.gca(projection='3d')
    ax = fig.add_subplot(111, projection='3d')
    samples_x = np.array([_.item() for _ in samples[:,0]])
    samples_y = np.array([_.item() for _ in samples[:,1]])
    lower = min(samples[:,0]).item(), min(samples[:,1]).item()
    upper = max(samples[:,0]).item(), max(samples[:,1]).item()
    plot_mean_and_95_2D(gp, lower, upper, ax, label=label )
    ax.scatter(samples_x, samples_y, rewards, c="r", marker='x', label="data")
    ax.scatter(samples_x, samples_y, prediction_points, c="b", marker='o', label="prediction")
    
    #spread = upper-lower
    plt.legend()
    ax.set_xlabel(dims[0]+" value")
    ax.set_ylabel(dims[1]+" value")
    ax.set_zlabel('Reward')
    plt.show()

def plot_samples_and_rewards_1D(samples, rewards, predictions, gp=None, column_index=0, label=""):
    offset_samples = samples[:]
    prediction_points = np.array([pt[0] for pt in predictions[0]])
    stds = np.array([pt for pt in predictions[1]])
    plt.scatter(offset_samples, rewards, color="r", label="data")
    plt.scatter(offset_samples, prediction_points, color="b", marker = '^', label="prediction")
    
    lower = min(offset_samples)
    upper = max(offset_samples)
    spread = upper-lower
    plot_mean_and_95(gp, lower-0.15*spread, upper+0.15*spread, label=label)
    plt.xlabel(label+" value")
    plt.ylabel("Reward")
    plt.legend()
    plt.show()

def plot_mean_and_95(gp, lower, upper, label=""):
    points =np.linspace(lower, upper, 200)
    points_vector = np.matrix(points).T
    mean, stdev = gp.predict(points_vector, return_std=True)
    plt.plot(points, mean, color="r", label=label)
    y_above = [mean[i].item()+1.96*stdev[i] for i  in range(mean.shape[0])]
    y_below = [mean[i].item()-1.96*stdev[i] for i  in range(mean.shape[0])]
    plt.fill_between(points, y_below, y_above, color="r", alpha=0.3)

def plot_mean_and_95_2D(gp, lower, upper, ax, label=""):
    num_points = 400
    xpoints =  np.linspace(lower[0], upper[0], num_points)
    ypoints =  np.linspace(lower[1], upper[1], num_points)
    points_vector = np.hstack([np.matrix(xpoints).T, np.matrix(ypoints).T])
    X, Y = np.meshgrid(xpoints, ypoints)
    Z = np.zeros(X.shape)
    for i in range(num_points):
        pred = gp.predict(np.hstack([np.matrix(X[i,:]).T, np.matrix(Y[i,:]).T]))
        Z[i,:] = pred.T
        
    mean, stdev = gp.predict(points_vector, return_std=True)
    #z_up = [mean[i].item()+1.96*stdev[i] for i  in range(mean.shape[0])]
    #z_down = [mean[i].item()-1.96*stdev[i] for i  in range(mean.shape[0])]
    #cset_up = ax.plot_surface(xpoints, ypoints, z_up, cmap=cm.coolwarm)
    cset_mean = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, antialiased=False, linewidth=0)
    #cset_down = ax.plot_surface(xpoints, ypoints, z_down, cmap=cm.coolwarm)
    

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
    return np.median(total_error)**0.5



def test_set(use_nn=False):
    print("Error", test(samples_test, use_nn=use_nn)[0])

def test_noise(use_nn=False):
    noise_levels = [0.8, 0.3, 0.15, 0.09, 0.05,0.01, 0.001, 0.0001, 0]
    noise_levels = [0]

    for noise_level in noise_levels:
         errors = []
         for i in range(15):
	     noisy_samples = add_noise(samples, noise_level)
             errors.append(test(noisy_samples, use_nn=use_nn)[0])
         print("noise_level", noise_level, "average noise", sum(errors)/len(errors))


def uniform_random_sample():
    lower = [-0.2,0,0.4, 2.51]
    upper = [0.2,0.8,4,2.51]
    sample = []
    for i in range(len(lower)):
        sample.append((upper[i] - lower[i]) * random()+ lower[i])
    return sample



def optimize(model, N = 5):
    #picks the value that's highest in the GP
    scores = []
    for i in range(N):
        random_action = uniform_random_sample()
        score, stdev = model.predict([random_action], return_std = True)
        scores.append(score)
    best_score_i = np.argmax(scores)
    best_score = scores[best_score_i]
    return best_score.item()

#test_set(use_nn=False)
#test_noise(use_nn=False)
#noisy_samples = add_noise(samples, 0.01)
#predictions = test(samples)[1]

#plot_samples_and_rewards_1D(samples, rewards, predictions)
#plot_samples_and_rewards(samples, rewards)
"""
Ns = [5, 10, 100, 500, 1000, 5000,10000,20000,30000,40000]
scores = []
for N in Ns:
    print("N = ", N)
    scores.append(optimize(gp, N))

print scores
pdb.set_trace()
plt.plot(Ns, scores)
plt.xlabel("N")
plt.ylabel("best score found")
plt.title("Effect of number of samples on the best score")
plt.show()
"""
#Needs to be vel, total, offset, or height
#for name in ["offset", "height", "total", "vel"]:
#    fit_and_test_one_dim(name)
def hyperparam_opt():
    samples, rewards = load_datasets("gp_learn_pour_and_grasp")
    lls, fits = [], []
    alphs = np.linspace(0, 10, 50)
    for alp in alphs:
	fit, ll = fit_and_test_n_dim(samples, rewards, 8, alpha=1, length_scale= alp)
	fits.append(fit)
	lls.append(ll)

def clean_rewards(rewards):
    cleaned_rewards = np.zeros(rewards.shape)
    for i in range(rewards.shape[0]):
        if rewards[i] == -30:
            cleaned_rewards[i] = 0
        else:
            cleaned_rewards[i] = rewards[i]
    return cleaned_rewards

samples, rewards = load_datasets("PR22D")
fit_and_test_two_dim("forward distance_height", samples_file = samples, rewards_file = rewards)
rewards = clean_rewards(rewards)
fit, ll = fit_and_test_n_dim(samples, rewards, 8,use_nn =False)
print("Fit", fit)
    
    
    

