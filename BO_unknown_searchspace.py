# -*- coding: utf-8 -*-
"""
@author: huongha
@description: Code implementation (for synthetic functions) of the Bayesian optimization framework with
			  unknown search space as proposed in the paper 'Bayesian Optimization with unknown
			  search space' (NeurIPS'19), Ha et al.
"""


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

import numpy as np
from numpy.linalg import pinv
from scipy.optimize import minimize

import functions
from utils.func_helpers import acq_maximize_fixopt, acq_gp, gram_matrix, acq_maximize_fixopt_local, acq_lcb
import time


# Declare the model one wants to investigate
bm_function = 'Hartman3'


# Load the synthetic functions
if (bm_function == 'Hartman3'):
    myfunction = functions.hartman_3d()
elif (bm_function == 'Hartman6'):
    myfunction = functions.hartman_6d()
elif (bm_function == 'Beale'):
    myfunction = functions.beale()
elif (bm_function == 'Ackley10'):
    myfunction = functions.ackley(10)
elif (bm_function == 'Levy3'):
    myfunction = functions.Levy(3)
elif (bm_function == 'Levy10'):
    myfunction = functions.Levy(10)
elif (bm_function == 'Eggholder'):
    myfunction = functions.egg_holder()
else:
    raise AssertionError("Unexpected value of 'bm_function'!")
func = myfunction.func
bounds_all = np.asarray(myfunction.bounds)
box_len = 0.2*np.max(bounds_all[:, 1] - bounds_all[:, 0]) # Box length to be 20% of the original box
print('Box length: {}'.format(box_len))


# Experiments with the proposed alogrithm
epsilon = 0.05 # Set the epsilon-accuracy condition
iter_mul = 10 # Set the evaluation budget, i.e. iter_mul x dim
max_exp = 30 # Set the number of experiments
Y_max_all = []
bounds_final_all = []
X_init_all = []
Y_init_all = []
gp_all = []
regret_all = []
r_optimal_all = []
time_all = []
for n_exp in range(max_exp):

    print('Test Function: ' + bm_function)
    print('Experiment {}:'.format(n_exp))

    # Set seed and initial points
    np.random.seed(n_exp)
    n_init_points = 3*myfunction.input_dim
    gp_exp = []

    # Randomly generate search space
    b_init_center = np.zeros((len(bounds_all), 1))
    for i in range(len(bounds_all)):
        b_init_center[i] = np.random.uniform(bounds_all[i][0]+box_len/2,
                                             bounds_all[i][1]-box_len/2)

    b_init_lower = b_init_center - box_len/2
    b_init_upper = b_init_center + box_len/2
    bounds_user = np.asarray([b_init_lower.ravel(), b_init_upper.ravel()]).T
    bounds = bounds_user.copy()
    b_init_center = np.mean(bounds, axis=1)
    print('Bounds: {}'.format(bounds))

    # Generate and normalize input, ouput
    temp = [np.random.uniform(x[0], x[1], size=n_init_points) for x in bounds]
    temp = np.asarray(temp)
    temp = temp.T
    X_init = list(temp.reshape((n_init_points, -1)))
    Y_init = func(X_init[0])
    for i in range(1, len(X_init)):
        temp = func(X_init[i])
        Y_init = np.append(Y_init, temp)
    Y = (Y_init-np.mean(Y_init))/(np.max(Y_init)-np.min(Y_init))

    # Start measure time
    start = time.time()

    # Fit Gaussian Process
    # Be careful with the fitting methodology (it can be stuck in local minima)
    # Multi-start?
    ls_prior = [0.01, 0.1, 1, 10]
    ml_value = np.zeros(len(ls_prior))
    gp_temp = []
    for idx, ls in enumerate(ls_prior):
        kernel = 1.0 * RBF(length_scale=ls)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=25)
        gp.fit(X_init, Y)
        gp_temp.append(gp)
        ml_value[idx] = gp.log_marginal_likelihood(gp.kernel_.theta)
    gp = gp_temp[np.min(np.argmax(ml_value))]
    gp_exp.append(gp)
    print('Kernel {} with Log Marginal Likelihood {}'.format(gp.kernel_, gp.log_marginal_likelihood(gp.kernel_.theta)))

    # Extract Gaussian Process hyper-parameters
    # kernel_k2 is the length-scale
    # theta_n is the scale factor
    kernel_k1 = gp.kernel_.get_params()['k1__constant_value']
    kernel_k2 = gp.kernel_.get_params()['k2__length_scale']
    theta_n = np.sqrt(kernel_k1)

    # Re-arrange the initial random observations
    max_iter = iter_mul*myfunction.input_dim
    Y_max = np.zeros((max_iter+n_init_points, 1))
    for i in range(n_init_points):
        Y_max[i] = np.max(Y_init[0:i+1])

    # Set some parameters for b_n and the evaluation budget
    max_iter_i = iter(range(1, max_iter+1))
    nu = 0.2 
    sigma = 0.1
    
    # List to store important values
    regret_iter = []
    r_optimal_iter = []
    bounds_iter = []
    iter_adjust = 0
    regret = 10*epsilon
    for n_iter_i in max_iter_i:

        n_iter = n_iter_i - iter_adjust
        print("Iteration {}".format(n_iter))

        # Compute the acquisition function at the observation points
        # b_n is found by using the correct formula
        lengthscale = gp.kernel_.get_params()['k2__length_scale']
        thetan_2 = gp.kernel_.get_params()['k1__constant_value']
        radius = np.abs(np.max(bounds[:, 1]-bounds[:, 0]))
        b = 1/2*np.sqrt(2)*np.sqrt(thetan_2)/lengthscale
        a = 1
        tau_n = (4*myfunction.input_dim+4)*np.log(n_iter) + 2*np.log(2*np.pi**2/(3*sigma)) \
                + 2*myfunction.input_dim*np.log(myfunction.input_dim*b*radius*np.sqrt(np.log(4*myfunction.input_dim*a/sigma)))
        b_n = np.sqrt(np.abs(nu*tau_n))
        y_init_mean, y_init_std = gp.predict(X_init, return_std=True)
        acq_init = y_init_mean.ravel() + b_n*y_init_std.ravel()
        print('b_n is: {}'.format(b_n))

        # Optimize the acquisition function
        x_max = acq_maximize_fixopt(gp, b_n, bounds)
        max_acq = acq_gp(x_max, gp, b_n)
        y_acq = acq_gp(np.asarray(X_init), gp, b_n)

        # Compute the maximum regret
        X_init_temp = X_init.copy()
        X_init_temp.append(x_max)
        Y_lcb_temp = acq_lcb(np.asarray(X_init_temp), gp, b_n)
        regret = max_acq - np.max(Y_lcb_temp)
        regret_iter.append(regret)
        print('Regret: {}'.format(regret))
        
        # Check if regret < 0, redo the optimization, typically redo the optimization
        # with starting point in X_init_temp
        indices_max_X = np.argsort(Y_lcb_temp)[::-1]
        n_search_X = np.min([20, len(indices_max_X)])
        if (regret + 1/n_iter**2 < 0):
            print('Regret < 0, redo the optimization')
            for i in range(n_search_X):
                x_try = X_init_temp[i]
                # Running L-BFGS-B from x_max
                res = minimize(lambda x: -acq_gp(x, gp, b_n),
                               x_try, bounds=bounds, method="L-BFGS-B")
            
                # Store it if better than previous minimum(maximum).
                if max_acq is None or -res.fun[0] >= max_acq:
                    x_max = res.x
                    max_acq = -res.fun[0]
                
                # Recompute regret
                regret = max_acq - np.max(Y_lcb_temp)
                print('Regret: {}'.format(regret))
            

        # Expand if regret < epsilon or the first iteration
        if (regret <= epsilon - 1/n_iter**2) | (n_iter_i == 1):
            print('Expanding bounds')

            # Expand the search space based on analytical formula in Theorem 1
            K = gram_matrix(X_init, 1, kernel_k2)
            K_inv = pinv(K)
            b = np.matmul(K_inv, Y)
            b_pos = b[b >= 0]
            b_neg = b[b <= 0]
            U, Sigma, V = np.linalg.svd(K_inv)
            lambda_max = np.max(Sigma)
            n = len(X_init)
            gamma = min(0.25*epsilon/max(np.sum(b_pos), -np.sum(b_neg)),
                        1/b_n*np.sqrt((0.5*epsilon*b_n*theta_n-0.0625*epsilon**2)/(n*lambda_max)))

            if (gamma > 1):
                raise ValueError('Gamma needs to be smaller than 1. Something wrong!')

            # Find d_{gamma}
            # For SE kernel, it's easy to find values of r to make k(r) < gamma
            scale_l = np.sqrt(-2*np.log(gamma/kernel_k1))
            print('The scale is: {}'.format(scale_l))

            bounds_ori_all = []
            for n_obs in range(len(acq_init)):
                X0 = X_init[n_obs]
                # Set the rectangle bounds
                bounds_ori_temp = np.asarray((X0 - scale_l*kernel_k2,
                                              X0 + scale_l*kernel_k2))
                bounds_ori = bounds_ori_temp.T
                bounds_ori_all.append(bounds_ori)

            bound_len = scale_l*kernel_k2
            temp = np.asarray(bounds_ori_all)
            temp_min = np.min(temp[:, :, 0], axis=0)
            temp_max = np.max(temp[:, :, 1], axis=0)
            bounds_ori = np.stack((temp_min, temp_max)).T
            bounds_new = bounds_ori.copy()
            print('Old bound: {}'.format(bounds))
            print('New bound: {}'.format(bounds_new))

            # Recompute bounds
            bounds = bounds_new.copy()
            bounds_iter.append(bounds)

            # Adjust iteration after adjusting bounds
            iter_adjust += (n_iter - 1)

            # Re-optimize the acquisition function with the new bound
            # and the new b_n
            n_iter = n_iter_i - iter_adjust 
            lengthscale = gp.kernel_.get_params()['k2__length_scale']
            thetan_2 = gp.kernel_.get_params()['k1__constant_value']
            radius = np.abs(np.max(bounds[:, 1]-bounds[:, 0]))
            b = 1/2*np.sqrt(2)*np.sqrt(thetan_2)/lengthscale
            a = 1
            tau_n = (4*myfunction.input_dim+4)*np.log(n_iter) + 2*np.log(2*np.pi**2/(3*sigma)) \
                    + 2*myfunction.input_dim*np.log(myfunction.input_dim*b*radius*np.sqrt(np.log(4*myfunction.input_dim*a/sigma)))
            b_n = np.sqrt(np.abs(nu*tau_n))

            x_max = acq_maximize_fixopt(gp, b_n, bounds)
            max_acq = acq_gp(x_max, gp, b_n)
            y_acq = acq_gp(np.asarray(X_init), gp, b_n)


            # Save some parameters of the bound
            X_init_bound = X_init.copy()
            Y_bound = Y.copy()
            lengthscale_bound = gp.kernel_.get_params()['k2__length_scale']
            scale_l_bound = scale_l

        # Check if the acquistion function argmax is at infinity
        # Then re-optimize within the bound that has the largest Y value
        if ((max_acq - b_n*np.sqrt(thetan_2)) >= -epsilon) & ((max_acq - b_n*np.sqrt(thetan_2)) <= 0):
            print('Re-optimize within smaller spheres')
            indices_max = np.argsort(Y_bound)[::-1]
            
            # Set a minimal number of local optimizations
            n_search = np.min([5, len(indices_max)])
            for i in range(n_search):
                X0 = X_init_bound[indices_max[i]]
                bounds_new = np.asarray((X0 - bound_len,
                                        X0 + bound_len))
                bounds_new = bounds_new.T
                x_max_local = acq_maximize_fixopt_local(gp, b_n, bounds_new, acq_type='ucb')
                max_acq_local = acq_gp(x_max_local, gp, b_n)
                max_inf = b_n*np.sqrt(gp.kernel_.get_params()['k1__constant_value'])
                if np.abs(max_acq_local-max_inf) <= epsilon:
                    x_max = x_max_local
                    break

        # Compute y_max
        y_max = func(x_max)
        Y_init = np.append(Y_init, y_max)
        X_init.append(x_max)
        Y = (Y_init-np.mean(Y_init))/(np.max(Y_init)-np.min(Y_init))

        # Update the kernel
        gp = GaussianProcessRegressor(kernel=kernel,
                                      n_restarts_optimizer=25)
        gp.fit(X_init, Y)
        gp_exp.append(gp)
        print('Kernel {} with Log Marginal Likelihood {}'.format(gp.kernel_, gp.log_marginal_likelihood(gp.kernel_.theta)))

        # Extract Gaussian Process hyper-parameters and update prior kernel
        # kernel_k2 is the length-scale
        # theta_n is the scale factor
        kernel_k1 = gp.kernel_.get_params()['k1__constant_value']
        kernel_k2 = gp.kernel_.get_params()['k2__length_scale']
        theta_n = np.sqrt(kernel_k1)
        kernel = kernel_k1*RBF(length_scale=kernel_k2)

        print('Maximum of y: {}'.format(np.max(Y_init)))
        Y_max[n_init_points+n_iter_i-1, 0] = np.max(Y_init)

    # End measuring time
    end = time.time()
    time_search_train = end-start
    print('Time cost (seconds): {:.2f}'.format(time_search_train))
    time_all.append(time_search_train)

    Y_max_all.append(Y_max)
    X_init_all.append(X_init)
    Y_init_all.append(Y_init)
    regret_all.append(regret_iter)
    r_optimal_all.append(r_optimal_iter)
    bounds_final_all.append(bounds_iter)

    # Save the list to files
    filename = 'results/result_' + bm_function + '_proposed_GPUBO_eps' +  str(epsilon).replace('.', '') + '.npz'
    np.savez(filename,
             *Y_max_all, *bounds_final_all, *X_init_all, *Y_init_all,
             *regret_all, *r_optimal_all, *bounds_final_all, *time_all)


 