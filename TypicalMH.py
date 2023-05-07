import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from array import array
import time
import scipy.stats as st
import seaborn as sns
import pandas as pd
import math
import pickle


# MH Method
def MH(z, T, theta_ini, logpior, loglik):
    N = z.shape[0]
    d = len(theta_ini)
    theta = np.zeros((d, T + 1))
    theta[:, 0] = theta_ini
    time_start = time.time()
    c = np.arange(N)
    for i in range(T):
        tp = np.random.normal(theta[:, i], 0.05, d)  # RW proposal to be tuned, and mean is theta，var is 0.05
        u = np.random.rand()  # Random number from 0 to 1
        # theta[:,i] is the current theta，tp is theta in the next step and is waiting to be decided to accept or reject
        start_logp = logpior(theta[:, i]) + np.sum(loglik(c, z, theta[:, i]), axis=0)
        new_logp = logpior(tp) + np.sum(loglik(c, z, tp), axis=0)
        energy_change = new_logp - start_logp

        if np.log(u) < energy_change:
            theta[:, i + 1] = tp
        else:
            theta[:, i + 1] = theta[:, i]

    time_end = time.time()
    time_sum = time_end - time_start  # Running time
    print(time_sum)
    return theta


# APMHT method
def APMHT(z, T, ep, m, theta_ini,logpior, loglik):
    N = z.shape[0]
    d = len(theta_ini)
    theta = np.zeros((d, T + 1))
    theta[:, 0] = theta_ini
    number_llik_eval = 0
    time_start = time.time()
    for i in range(T):
        tp = np.random.normal(theta[:, i], 0.05, d)
        u = np.random.rand()
        l = 0
        lbar = 0
        lsqbar = 0
        n = 0
        done = False
        # calculate mu0
        mu0 = (1 / N) * (np.log(u) + logpior(theta[:, i]) - logpior(tp))
        # set batch
        batch = 0
        # index of samples
        index = np.arange(N)
        while not done:
            draw = np.random.randint(0, len(index), min(m, N - n))
            batch = np.hstack((draw, batch))
            n = n + min(m, N - n)
            # sampling without replacement, so drop the columns in draw
            index = np.delete(index, draw)
            l = loglik(batch, z, tp) - loglik(batch, z, theta[:, i])
            # mean of the columns
            lbar = np.mean(l, axis=0)
            lsqbar = np.mean(l ** 2, axis=0)

            sd_batch = (n / (n - 1)) ** (1 / 2) * (lsqbar - lbar ** 2)
            sd_hat = (sd_batch / (n ** (1 / 2))) * ((1 - (n - 1) / (N - 1)) ** (1 / 2))

            # t test×2
            delta = st.t.sf(abs((lbar - mu0) / sd_hat), n - 1) * 2

            if delta < ep:
                if lbar > mu0:
                    theta[:, i + 1] = tp
                else:
                    theta[:, i + 1] = theta[:, i]
                done = True
                number_llik_eval = number_llik_eval + n

    time_end = time.time()
    time_sum = time_end - time_start  # Running time
    print(time_sum)
    print(number_llik_eval)
    return theta, number_llik_eval


# Confidence Sampler
## Concentration bounds
# N:Number of points

def ctBernsteinSerfling(N, n, a, b, sigma, delta):
    """
    Bernstein-type bound without replacement, from (Bardenet and Maillard, to appear in Bernoulli)
    """
    l5 = np.log(5 / delta)
    kappa = 7.0 / 3 + 3 / np.sqrt(2)
    if n <= N / 2:
        rho = 1 - 1.0 * (n - 1) / N
    else:
        rho = (1 - 1.0 * n / N) * (1 + 1.0 / n)
    return sigma * np.sqrt(2 * rho * l5 / n) + kappa * (b - a) * l5 / n


def ctHoeffdingSerfling(N, n, a, b, delta):
    """
    Classical Hoeffding-type bound without replacement, from (Serfling, Annals of Stats 1974)
    """
    l2 = np.log(2 / delta)
    if n <= N / 2:
        rho = 1 - 1.0 * (n - 1) / N
    else:
        rho = (1 - 1.0 * n / N) * (1 + 1.0 / n)
    return (b - a) * np.sqrt(rho * l2 / 2 / n)


def ctBernstein(N, n, a, b, sigma, delta):
    """
    Classical Bernstein bound, see e.g. the book by Boucheron, Lugosi, and Massart, 2014.
    """
    l3 = np.log(3 / delta)
    return sigma * np.sqrt(2 * l3 / n) + 3 * (b - a) * l3 / n


# Confidence Sampler method
def Conf(z, T, m, theta_ini,logpior,loglik):
    N = z.shape[0]
    d = len(theta_ini)
    theta = np.zeros((d, T + 1))
    theta[:, 0] = theta_ini
    number_llik_eval = 0
    delta = .1
    gamma = 2.
    time_start = time.time()
    for i in range(T):
        np.random.shuffle(z)
        tp = np.random.normal(theta[:, i], 0.05, d)
        u = np.random.rand()
        l = 0
        al = np.arange(N)
        lall = loglik(al, z, tp) - loglik(al, z, theta[:, i])
        a = np.min(lall)
        b = np.max(lall)
        n = m
        done = False

        psi = (1 / N) * (np.log(u) + logpior(theta[:, i]) - logpior(tp))

        batch = 0

        index = np.arange(N)
        cpt = 0
        while not done and n < N:
            n = min(N, np.floor(gamma * n))
            batch = np.arange(int(n))
            cpt += 1
            deltaP = delta / 2 / cpt ** 2
            # print(batch)
            l = loglik(batch, z, tp) - loglik(batch, z, theta[:, i])

            Lambda = np.mean(l)
            sigma = np.std(l)

            c = ctBernstein(N, n, a, b, sigma, deltaP)

            if np.abs(Lambda - psi) > c:
                done = True

        if Lambda > psi:
            theta[:, i + 1] = tp
        else:
            theta[:, i + 1] = theta[:, i]

        number_llik_eval = number_llik_eval + n

    time_end = time.time()
    time_sum = time_end - time_start
    print(time_sum)
    print(number_llik_eval)
    return theta, number_llik_eval

file = open('Xc_4000.pickle', 'rb') #二进制读取
Xc_4000 = pickle.load(file) #读取结果存入a_dict1
file.close()
# MinibatchMH Method
def MiniMH(z, T, m, theta_ini,logpior, loglik):
    N = z.shape[0]
    d = len(theta_ini)
    theta = np.zeros((d, T + 1))
    theta[:, 0] = theta_ini
    number_llik_eval = 0
    full_N = 0
    num_Xc = 0
    time_start = time.time()
    for i in range(T):
        tp = np.random.normal(theta[:, i], 0.05, d)
        u = np.random.rand()
        n = 0
        done = False

        batch = -1

        index = np.arange(N)
        psi = logpior(theta[:, i]) - logpior(tp)
        while not done:
            draw = np.random.randint(0, len(index), min(m, N - n))
            if type(batch) == int:
                batch = draw
            else:
                batch = np.hstack((draw, batch))
            n = n + min(m, N - n)

            index = np.delete(index, draw)
            l = (N / 10) * (loglik(batch, z, tp) - loglik(batch, z, theta[:, i]))

            # 对数似然比求均值按列
            #             sumOfSquares=np.sum(l**2,axis=0)
            #             sumOfValues=np.sum(l,axis=0)

            deltaStar = np.mean(l, axis=0) - psi
            #             sampleVariance = (sumOfSquares/n - (sumOfValues/n)**2)/ n
            sampleVariance = (1 / n) * np.var(l, axis=0)

            #             print("sampleVariance",sampleVariance)
            #             print("sampleVariance2",sampleVariance2)
            # print(sampleVariance)

            numStd = deltaStar / np.sqrt(sampleVariance)
            #             print(deltaStar)
            #             print(numStd)
            accept = False
            if n == N:
                full_N += 1
                # print("WARNING: test used entire dataset but variance is still too high.")
                if deltaStar > 0:
                    accept = True
            # Abnormally good or bad minibatches.
            elif abs(numStd) > 5:
                if numStd > 0:
                    accept = True
                else:
                    continue
            elif sampleVariance >= 1:
                # print("WARNING: test used entire dataset but variance is still too high.")
                continue
            else:
                Xn = np.random.normal(0, 1 - sampleVariance, 1)
                Xc = np.random.choice(Xc_4000)
                testStat = deltaStar + Xn + Xc
                num_Xc += 1
                if testStat > 0:
                    accept = True

            if accept:
                theta[:, i + 1] = tp
            else:
                theta[:, i + 1] = theta[:, i]
            done = True
            number_llik_eval = number_llik_eval + n

    time_end = time.time()
    time_sum = time_end - time_start  # Running time
    print(time_sum)
    print("full_N_time:", full_N)
    print("num_Xc:", num_Xc)
    return theta, number_llik_eval