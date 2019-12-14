
import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.style.use('seaborn-darkgrid')


# # Initialize random number generator
# np.random.seed(123)

# True parameter values
alpha = 3
sigma = 1
beta = [1, 0.5]

# Size of dataset
size = 100
#
nsamples = 1000

# Predictor variable
X1 = np.random.randn(size)
X2 = np.random.randn(size) 
epsilon = np.random.randn(size)

# Simulate outcome variable
Y = alpha + beta[0]*X1 + beta[1]*X2 + epsilon*sigma


X = np.linspace(0, 1, size)
# # y = a + b*x
# def glm(alpha, beta0, beta1, X):
#     Y_true = alpha + beta[0]*X + beta[1]*X
#     return Y_true

# Y = glm(alpha, beta[0], beta[1], X)

glm_likelihood = pm.Model()
with glm_likelihood:

    # Priors for unknown model parameters
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=10, shape=2)
    sigma = pm.HalfNormal('sigma', sigma=1)

    # Expected value of outcome
    mu = alpha + beta[0]*X1 + beta[1]*X2

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=Y)

    # draw posterior samples
    trace = pm.sample(nsamples)

alpha_post = trace.get_values('alpha')
beta0_post = trace.get_values('beta')[:,0]
beta1_post = trace.get_values('beta')[:,1]
# Y_post = trace.get_values('Y_obs')

coefs_post = pd.DataFrame(data={'alpha':alpha_post, 'beta0':beta0_post, 'beta1':beta1_post})
preds = coefs_post.apply(lambda r : r['alpha'] + r['beta0']*X + r['beta1']*X, axis=1)

for p in preds[:100]:
    plt.plot(X, p, 'b-', alpha=0.02)
plt.scatter(X, Y)
plt.show()


# sns.jointplot(data = dats['glm'])


# map_estimate = pm.find_MAP(model=glm_likelihood)

# # pm.traceplot(trace)

# plt.figure(figsize=(7, 7))
# plt.plot(X, Y, 'x')
# pm.plot_posterior_predictive_glm(trace, samples=100)

# plt.show()
