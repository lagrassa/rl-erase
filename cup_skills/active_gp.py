# Author: Zi Wang
import numpy as np
import os
import GPy as gpy
import ipdb as pdb
from scipy.stats import norm
import helper
from active_learner import ActiveLearner
import time

class ActiveGP(ActiveLearner):
    '''
    Active learner with a GP backend.
    '''
    def __init__(self, func, initx, inity, query_type, flag_lk=False, 
                 is_adaptive=False, task_lengthscale=None, betalambda=0.999, 
                 sample_time_limit=200):
        '''
        func: scoring function
        initx: initial inputs
        inity: initial outputs
        query_type: type of query ('lse' or 'best_prob')
        flag_lk: False if using diverse sampler with a fixed kernel; 
                 True if using diverse sampler with a kernel learned online.
        is_adaptive: True if using the adaptive sampler; False if using the 
        diverse sampler; None if using rejection sampler with uniform proposal
        distribution.
        task_lengthscale: the inverse length scale of the kernel for diverse sampling.
        betalambda: a hyper parame
        sample_time_limit: time limit (seconds) for generating samples with (adaptive) 
        rejection sampling.
        '''
        if initx is not None and inity is not None:
	    if inity.ndim == 1:
	        inity = inity[:, None]
            self.xx = initx
            self.yy = inity
            assert(initx.ndim == 2 and inity.ndim == 2)
            self.init_len = len(initx)

        self.yvals = []
        self.func = func
        self.query_type = query_type
        self.name = 'gp_{}'.format(query_type)
        self.sampled_xx = []
        self.good_samples = []
        if task_lengthscale is None:
            self.task_lengthscale = func.task_lengthscale[func.param_idx]
        else:
            self.task_lengthscale = task_lengthscale
        self.beta = 3.24
        self.flag_lk = flag_lk
        self.is_adaptive = is_adaptive
        self.betalambda = betalambda
        self.sample_time_limit = sample_time_limit
    def query_best_prob(self, context):
        '''
        Returns the input that has the highest probability to be in the super 
        level set for a given context.
        '''
        x0, x0context = helper.find_closest_positive_context_param(
            context, self.xx, self.yy, self.func.param_idx, self.func.context_idx)
        self.model = self.model
        total_context = helper.tuple_context_to_total_context(context)
        def ac_f(x):
            if x.ndim == 1:
                x = x[None, :]
            x = np.hstack((x, np.tile(total_context, (x.shape[0], 1))))
            mu, var = self.model.predict(x)
            return (-mu)/np.sqrt(var)

        def ac_fg(x):
            if x.ndim == 1:
                x = x[None, :]
            x = np.hstack((x, np.tile(total_context, (x.shape[0], 1))))
            mu, var = self.model.predict(x)
            dmdx, dvdx = self.model.predictive_gradients(x)
            dmdx = dmdx[0, :, 0]
            dvdx = dvdx[0, :]
            f = (-mu)/np.sqrt(var)
            g = (-np.sqrt(var)*dmdx - 0.5*(-mu)*dvdx/np.sqrt(var)) / var
            return f[0, 0], g[0, self.func.param_idx]

        x0 = np.vstack((x0, self.xx[np.squeeze(self.yy)>0][:,self.func.param_idx]))
        x_star, y_star = helper.global_minimize(
            ac_f, ac_fg, self.func.x_range[:, self.func.param_idx], 10000, x0)
        print 'best beta=', -y_star
        self.best_beta = -y_star
        self.beta = norm.ppf(self.betalambda*norm.cdf(self.best_beta))
        if self.best_beta < 0:
            print('Warning! Cannot find any parameter to be super level set \
                   with more than 0.5 probability. Are you sure to continue?')
        if self.beta > self.best_beta:
            raise ValueError('Beta cannot be larger than best beta.')
        return np.hstack((x_star, total_context))

    def gen_adaptive_samples(self, context, n=10000, m=50):
     
        '''
        Generate adaptive samples with rejection sampling, where the proposal 
        distribution is uniform and truncated Gaussian mixtures.
        Args:
            context: the context the generator is conditioned upon.
            n: number of proposals per iteration.
            m: minimum number of samples to be generated.
        '''
	total_context = helper.tuple_context_to_total_context(context)
	def ac_f(x):
	    if x.ndim == 1:
		x = x[None, :]
	    x = np.hstack((x, np.tile(total_context, (x.shape[0], 1))))
	    mu, var = self.model.predict(x)
	    ret = (mu)/np.sqrt(var)
	    return ret.T[0]
	dx = len(self.func.param_idx)
	good_samples = np.zeros((0, dx))
	prob = np.zeros(0)
	t_start = time.time()
	sampled_cnt = 0

	xmin = self.func.x_range[0, self.func.param_idx]
	xmax = self.func.x_range[1, self.func.param_idx]
	prob_unif_unit = np.prod(xmax - xmin)
	x_samples = self.xx[np.squeeze(self.yy)>0][:,self.func.param_idx]
	if len(self.sampled_xx) == 0:
	    best_x = self.query_best_prob(context)
	    x_samples = np.vstack((x_samples, best_x[self.func.param_idx]))
	else:
	    x_samples = np.vstack((x_samples, self.sampled_xx[:, self.func.param_idx]))
	assert(len(x_samples) > 0)
	good_inds = ac_f(x_samples) > self.beta
	good_samples = np.vstack((x_samples[good_inds], good_samples))
	prob = np.hstack(( np.ones(len(good_samples)), prob ))
	scale = np.array(self.model.kern.lengthscale[self.func.param_idx]) * 1.
	flag = True
	while flag or len(good_samples) <= m:
	    flag = False # make sure it samples at least once
	    if time.time() - t_start > self.sample_time_limit:
		print('Elapsed sampling time = {}, sampling iterations = {}'.format(time.time() - t_start), sampled_cnt)
		raise ValueError('Not enough good samples.')
	    sampled_cnt += 1

	    x_samples_unif = np.random.uniform(xmin, xmax, (n, dx))
	    prob_unif = np.ones(n) * prob_unif_unit
	    good_inds = ac_f(x_samples_unif) > self.beta
	    x_samples_unif = x_samples_unif[good_inds]
	    prob_unif = prob_unif[good_inds]
	    good_samples = np.vstack((x_samples_unif, good_samples))
	    prob = np.hstack((prob_unif, prob))

	    if len(x_samples) > 0 and self.is_adaptive is not None:
		try:
		    x_samples_gmm, prob_gmm = helper.sample_tgmm(x_samples.astype(np.float32), scale, n, xmin, xmax)
		    good_inds = ac_f(x_samples_gmm) > self.beta
		    x_samples_gmm = x_samples_gmm[good_inds]
		    prob_gmm = prob_gmm[good_inds]
		    good_samples = np.vstack((x_samples_gmm, good_samples))
		    prob = np.hstack((prob_gmm, prob))
		    if len(x_samples_gmm) > n/2.:
			scale *= 2
		    elif len(x_samples_gmm) < n/10.:
			scale *= 0.5
		except:
		    print("TGMM sampling failed!!! Should investigate")
		    continue

	    if len(good_samples) < m:
		x_samples = good_samples
	    else:
		x_samples_inds = np.random.choice(np.arange(len(good_samples)), size=m, replace=False, p=prob/np.sum(prob))
		x_samples = good_samples[x_samples_inds]

        print('{} samples are generated with the adaptive sampler.'.format(len(good_samples)))
        self.good_samples = good_samples
        sample_diversity = helper.diversity(good_samples, self.func.param_idx, diverse_dims = self.func.diversity_important)
        return x_samples, sample_diversity

    def reset_sample(self):
        '''
        Clear the list of samples.
        '''
        self.sampled_xx = []
        self.good_samples = []

    def sample_adaptive(self, context):
        '''
        Returns one sample from the high probability super level set for a given context,
        using the adaptive sampler.
        '''
        total_context = helper.tuple_context_to_total_context(context)
    
        if len(self.sampled_xx) == 0:
            xx, sample_diversity = self.gen_adaptive_samples(context)
            self.unif_samples = np.hstack((xx, np.tile(total_context, (xx.shape[0], 1))))
            self.sampled_xx = np.array([self.unif_samples[0]])
        else:
            if len(self.unif_samples) < 10: 
                xx, sample_diversity = self.gen_adaptive_samples(context)
                self.unif_samples = np.hstack((xx, np.tile(total_context, (xx.shape[0], 1))))

            new_s = self.unif_samples[0]

            self.sampled_xx = np.vstack((self.sampled_xx, new_s))
        
        self.unif_samples = np.delete(self.unif_samples, (0), axis=0)
        
        return self.sampled_xx[-1]
    def sample(self, context):
        '''
        Returns one sample from the high probability super level set for a given context.
        '''
        if self.model is None:
            print("Warning! model uninitialized. active_learn before sampling")
        if self.is_adaptive:
            return self.sample_adaptive(context)

        if len(self.sampled_xx) == 0:
            self.sampled_xx = np.array([self.query_best_prob(context)])
        else:
            # Learning task-level kernel lengthscale
            if self.flag_lk and len(self.sampled_xx) >= 2:
                d = helper.important_d(self.sampled_xx[-1, self.func.param_idx], self.sampled_xx[:-1, self.func.param_idx], self.task_lengthscale)
                self.task_lengthscale[d] *= 0.7
            # End of learning task-level kernel lengthscale
            if len(self.good_samples) < 10:
                self.gen_adaptive_samples(context)
            sid = helper.argmax_condvar(self.good_samples, self.sampled_xx[:, self.func.param_idx], self.task_lengthscale)
            total_context = helper.tuple_context_to_total_context(context)
            new_s = np.hstack((self.good_samples[sid], total_context))
            self.good_samples = np.delete(self.good_samples, (sid), axis=0)
            self.sampled_xx = np.vstack((self.sampled_xx, new_s))
        sample_diversity = helper.diversity(self.sampled_xx,self.func.param_idx)
        return self.sampled_xx[-1], sample_diversity
   
    def query(self, context):
        '''
        Select the next input to query.
        '''
        if self.query_type is 'best_prob':
            return self.query_best_prob(context)
        elif self.query_type is 'lse':
            return self.query_lse(context)

    def retrain(self, newx=None, newy=None):
        '''
        Train the GP on all the training data again.
        '''
        if newx is not None and newy is not None:
            self.xx = np.vstack((self.xx, newx))
            self.yy = np.vstack((self.yy, newy))
            self.yvals.append(newy)
        lengthscale = (self.func.x_range[1] - self.func.x_range[0]) * 0.05
        if len(self.func.discrete_contexts) > 0:
            num_discrete = len(self.func.discrete_contexts[0])
            for i in range(num_discrete):
                only_that_param = [ctx[i] for ctx in self.func.discrete_contexts ]
                min_param = min(only_that_param)
                max_param = max(only_that_param)
                lengthscale = np.hstack([lengthscale, (max_param-min_param)*0.05])

        else:
            num_discrete = 0
 
        k = gpy.kern.Matern52(self.func.x_range.shape[1]+num_discrete, ARD=True, lengthscale=lengthscale)
        self.model = gpy.models.GPRegression(self.xx, self.yy, k)
        for i in range(self.func.dx):
            self.model.kern.lengthscale[i:i+1].constrain_bounded(self.func.lengthscale_bound[0][i],
                self.func.lengthscale_bound[1][i], warning=False)
        self.model['.*variance'].constrain_bounded(1e-1,2., warning=False)
        self.model['Gaussian_noise.variance'].constrain_bounded(1e-4,0.01, warning=False)
        # These GP hyper parameters need to be calibrated for good uncertainty predictions.
        self.model.optimize(messages=False)
        print self.model

    def restore_model(self, EXP_NAME=None):
        found_xx = False
        found_yy = False
        for f in os.listdir("data/"):
	    if EXP_NAME in f and "xx" in f:
	        xx = np.load("data/"+f)
	        found_xx = True
	    if EXP_NAME in f and "yy" in f:
	        yy = np.load("data/"+f)
	        found_yy = True
        if found_xx and found_yy:
	    print("Loading model...")
	    self.xx = xx
	    self.yy = yy.reshape(-1,1)

    def query_lse(self, context):
        '''
        Returns the next active query on the function in a particular context 
        using level set estimation.
        We here implement the straddle algorithm from 
        B. Bryan, R. C. Nichol, C. R. Genovese, J. Schneider, C. J. Miller, and L. Wasserman, 
        "Active learning for identifying function threshold boundaries," in NIPS, 2006.
        '''
        x0, x0context = helper.find_closest_positive_context_param(
            context, self.xx, self.yy, self.func.param_idx, self.func.context_idx)

        total_context = helper.tuple_context_to_total_context(context)
        def ac_f(x):
            if x.ndim == 1:
                x = x[None, :]
            x = np.hstack((x, np.tile(total_context, (x.shape[0], 1))))
            mu, var = self.model.predict(x)
            return -1.96*np.sqrt(var) + np.abs(mu)
        x_star, _ = helper.global_minimize(
            ac_f, None, self.func.x_range[:, self.func.param_idx], 10000, x0)
        
        return np.hstack((x_star, total_context))

