# Master-Thesis
A Minibatch MCMC Method Based on Pseudo-Marginal Approach and Control Variates

## Explanatory Note
There is a brief explanatory note for the code in the thesis.
In order to compare the results between out method PMMIN method and other four kind of MH method, we design four experiments in the thesis.

**TypicalMH.py**

There are four typical subsample MH methods.
* MH 
>Metropolis N, Rosenbluth A W, Rosenbluth M N, et al. Equation of state calculations by fast computing machines[J]. The Journal of Chemical Physics, 1953, 21(6): 10871092.

>Hastings W K. Monte Carlo sampling methods using Markov chains and their applications[J]. Biometrika, 1970, 57(1): 97109.
* APMHT 
>Korattikara A, Chen Y, Welling M. Austerity in MCMC land: Cutting the Metropolis-Hastings budget[C]. International Conference on Machine Learning. PMLR, 2014: 181189.
* Conf 
>Bardenet R, Doucet A, Holmes C. Towards scaling up Markov chain Monte Carlo: an adaptive subsampling approach[C]. International conference on Machine Learning. PMLR, 2014: 405413.
* MiniMH 
>Seita D, Pan X, Chen H, et al. An efficient minibatch acceptance test for MetropolisHastings[C]. Proceedings of the 27th International Joint Conference on Artificial Intelligence. 2018: 53595363.

**PMMINMethod.py**

PMMIN Method is the minibatch MCMC method based on Pseudo-Marginal approach and control variates.

**GenerateXc.ipynb**

Generating the additive variable in the MiniMH method and PMMIN method, the details about how to generate can be found in Seita's works.



**Exp1.ipynb**

The prior distribution of $\theta$ is a normal distribution whose mean is 1 and variance is 1.
The density function of samples is Gaussian and the mean is $\theta$ which needs to be estimated by Bayesian inference.
We generate $n$ samples with the true value of $\theta$ is 1, and generate the posterior distribution by these five methods.


**Exp2.ipynb**

$p(\theta) \sim N(0,diag(1,1)),$

$p(x_i \mid \theta) \sim 0.5\times N(\theta_1,1)+0.5\times N(\theta_1+\theta_2,1).$

Where $p(\theta)$ is the prior distribution, and $p(x_i \mid \theta)$ is the density function. We generete the samples with $\theta=(0,0.4)$. It is obvious that the samples are same as with $theta=(0.4,-0.4)$. So there are two peaks in posterior distribution.


**Exp3.ipynb**

$p(\theta) \sim N(0,I_d)$,

$z_i \sim B(\pi_i)$,

$\pi_i = \frac{e^{\theta_1 x_{1i}+\dots+\theta_d x_{di}}}{1+e^{\theta_1 x_{1i}+\dots+\theta_d x_{di}}}$.


Where $p(\theta)$ is the prior distribution. We let $d=20$, $x_i=(x_{1i},\dots, x_{20i})$ that are generated independently from a standard Normal distribution.





