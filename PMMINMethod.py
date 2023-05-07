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


def dist_eucl(x1, x2):
    # 算出1行d列个距离
    return np.linalg.norm(x1 - x2, axis=1)


def clustering(x, eps):
    n = x.shape[0]
    I = np.zeros(n)  # Indicator variable, 1 if obs i has been given a cluster
    z = np.zeros(x.shape[1])  # centroids
    C = np.zeros(1)  # indices inside each clusters
    k = 0
    index = np.arange(n)
    for j in range(n):
        if I[j] == 0:
            p = np.where(dist_eucl(x[np.array(1 - I, dtype=bool), :], x[j, :]) <= eps)
            C_current = index[p]

            index = np.delete(index, p)

            if len(C_current) == 1:
                z = np.vstack((x[C_current, :][0], z))
            else:
                z = np.vstack(((1 / len(C_current)) * np.sum(x[C_current, :], axis=0), z))
            I[C_current] = 1
            C = pd.concat([pd.DataFrame(C_current), pd.DataFrame(C)], axis=1)
            k = k + 1
    return k, z.transpose(), C


def generateB(data,outk,outz,C):
    B=[]
    d=outz.shape[0]
    for k in range(outk):
        B.append(np.zeros((d,d)))
        for i in C.iloc[:,k]:
            if not math.isnan(i):
                b=data[int(i),:]-outz[:,k]
                m=b.shape[0]
                B[-1]=B[-1]+np.dot(b.reshape(m,1),b.reshape(1,m))
    return B

file = open('Xc_4000.pickle', 'rb') #二进制读取
Xc_4000 = pickle.load(file) #读取结果存入a_dict1
file.close()

def PMMIN(data, z, K, C, m, Time, theta_ini,B, logprior,loglik,gradient_llik,hessian_llik):
    # z是质心
    # N个样本点
    N = data.shape[0]
    d = len(theta_ini)
    d_b = np.zeros(m)
    d_bp = np.zeros(m)
    theta = np.zeros((d, Time + 1))
    theta[:, 0] = theta_ini
    Xc_num = 0
    time_start = time.time()

    for t in range(Time):
        accept = False
        bp = np.random.normal(theta[:, t], 0.05, d)  # RW proposal to be tuned
        u = np.random.choice(np.arange(N), m, replace=True)  # 放回抽样？抽m个样本
        for i in range(m):
            # 找出这个样本属于的类别
            coord = np.where(C.values == u[i])[1][0]
            # print(coord)
            d_b[i] = loglik(data[u[i], :], theta[:, t]) - loglik(z[:, coord], theta[:, t]) - np.dot(
                gradient_llik(z[:, coord], theta[:, t]), (data[u[i], :] - z[:, coord])) - 0.5 * (
                                 data[u[i], :] - z[:, coord]).dot(hessian_llik(z[:, coord], theta[:, t])).dot(
                data[u[i],] - z[:, coord])
            d_bp[i] = loglik(data[u[i], :], bp) - loglik(z[:, coord], bp) - np.dot(gradient_llik(z[:, coord], bp), (
                        data[u[i], :] - z[:, coord])) - 0.5 * (data[u[i], :] - z[:, coord]).dot(
                hessian_llik(z[:, coord], bp)).dot(data[u[i],] - z[:, coord])
            # print(d_b[i])
        #print(d_b)
        mu_b = np.mean(d_b)
        sigma_b = np.var(d_b)
        #print(sigma_b)
        mu_bp = np.mean(d_bp)
        sigma_bp = np.var(d_bp)
        first_term_b = 0
        first_term_bp = 0
        third_term_b = 0
        third_term_bp = 0
        for k in range(K):
            Nk = C.iloc[:, k].count()
            first_term_b = first_term_b + Nk * loglik(z[:, k], theta[:, t])
            first_term_bp = first_term_bp + Nk * loglik(z[:, k], bp)
            H_bp = hessian_llik(z[:, k], bp)
            H_b = hessian_llik(z[:, k], theta[:, t])
            third_term_b = third_term_b + H_b * B[k]
            third_term_bp = third_term_bp + H_bp * B[k]

        q_b = first_term_b + 0.5 * sum(third_term_b.sum(axis=1))
        q_bp = first_term_bp + 0.5 * sum(third_term_bp.sum(axis=1))

        l_hat_b = q_b + N * mu_b - (N ** 2) / (2 * m) * sigma_b
        l_hat_bp = q_bp + N * mu_bp - (N ** 2) / (2 * m) * sigma_bp

        prob = (l_hat_bp + logprior(bp)) - (l_hat_b + logprior(theta[:, t]))

        sampleVariance = (N ** 2 / m) * (sigma_b + sigma_bp + 2 * (sigma_b * sigma_bp) ** 0.5)
        # print("sampleVariance:",sampleVariance)
        accept = False
        if sampleVariance < 1:
            Xn = np.random.normal(0, 1 - sampleVariance, 1)
            Xc = np.random.choice(Xc_4000)
            testStat = prob + Xn + Xc
            # testStat=numStd+Xc
            Xc_num += 1
            if testStat > 0:
                accept = True
        else:
            if prob >= np.log(np.random.rand()):
                accept = True

        # p_accept = np.exp(prob)/(1+np.exp(prob))

        if accept:
            theta[:, t + 1] = bp
        else:
            theta[:, t + 1] = theta[:, t]
        # if t % 1000 == 0:
        #     print(t // 1000)

    time_end = time.time()
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    print(time_sum)
    # print(Xc_num)
    return theta

