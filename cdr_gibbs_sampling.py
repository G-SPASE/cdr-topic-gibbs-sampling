#!/usr/bin/env python
# encoding: utf-8

import time
import uuid
import numpy as np
import csv


T = 24
D = 51
K = 5
MAX_ITER = 30

alpha = 0.5
beta = 0.5
gamma = 0.5

users = dict({})


def read_tower_index(filename):
    tower_index = dict({})
    tower_idx2id = dict({})
    with open(filename, 'r') as f:
        for tidx_str, tid_str, lat_str, lon_str in csv.reader(f):
            tidx = int(tidx_str)
            tid = int(tid_str)
            lat = float(lat_str)
            lon = float(lon_str)
            tower_index[tid] = [tidx, lat, lon]
            tower_idx2id[tidx] = tid

    return tower_index, tower_idx2id


def get_sparse_affinity(tower_index, tower_idx2id, tower_graph):
    affinity_sparse_matrix = dict({})
    for tid1 in tower_index:

        tidx1 = tower_index[tid1][0]
        weight = tower_graph[:, tidx1]
        weight /= np.sum(weight)

        for tidx2 in list(np.where(weight>0.0))[0]:
            # tid2 = tower_idx2id[tidx2]
            if tidx1 not in affinity_sparse_matrix:
                affinity_sparse_matrix[tidx1] = [(tidx2, weight[tidx2])]
            else:
                affinity_sparse_matrix[tidx1].append((tidx2, weight[tidx2]))

    return affinity_sparse_matrix


def sample_index(p):
    return np.random.multinomial(1, p).argmax()


def conditional_distribution(ntk, ndk, nulk, nlk, nuk, nk, nu, u, d, t, l):
    p_theta = (nuk[u, :] + alpha) / (nu[u] + alpha)
    p_t = (ntk[t, :] + beta) / (nk + beta)
    p_d = (ndk[d, :] + beta) / (nk + beta)
    p_phi_l = (nulk[u, l, :] + gamma) / (nuk[u, :] + gamma)
    #  p_phi_g = (nlk[l, :] + gamma) / (nk + gamma)

    p_k = p_theta * p_t * p_d * p_phi_l
    p_k /= np.sum(p_k)
    return p_k


def crowdtopic(data, tower_index, affinity_sparse_matrix):
    U = len(users)
    L = len(tower_index)
    N = len(data)
    print 'Number of Users: {}\tNumber of Towers: {}\tNumber of Records: {}\n'.format(U, L, N)

    topics = np.zeros(N, dtype=np.int32)

    ntk = np.zeros((T, K), dtype=np.int32)
    ndk = np.zeros((D, K), dtype=np.int32)
    nulk = np.zeros((U, L, K), dtype=np.float)
    nlk = np.zeros((L, K), dtype=np.float)
    nuk = np.zeros((U, K), dtype=np.int32)
    nk = np.zeros(K, dtype=np.int32)
    nu = np.zeros(U, dtype=np.int32)

    for i in xrange(N):
        u, d, t, l = data[i]
        k = np.random.randint(K)
        topics[i] = k
        ntk[(t, k)] += 1
        ndk[(d, k)] += 1
        for tidx_a, w_a in affinity_sparse_matrix[l]:
            nulk[(u, tidx_a, k)] += w_a
            nlk[(tidx_a, k)] += w_a
        nk[k] += 1
        nu[u] += 1
    print 'Initialization Complete'

    for i in xrange(MAX_ITER):
        print 'Iteration {}'.format(i)
        for j in xrange(N):
            u, d, t, l = data[j]
            k = topics[j]
            ntk[(t, k)] -= 1
            ndk[(d, k)] -= 1
            for tidx_a, w_a in affinity_sparse_matrix[l]:
                nulk[(u, tidx_a, k)] -= w_a
                nlk[(tidx_a, k)] -= w_a
            nuk[(u, k)] -= 1
            nk[k] -= 1

            p_k = conditional_distribution(ntk, ndk, nulk, nlk, nuk, nk, nu, u, d, t, l)
            k = sample_index(p_k)

            topics[j] = k
            ntk[(t, k)] += 1
            ndk[(d, k)] += 1
            for tidx_a, w_a in affinity_sparse_matrix[l]:
                nulk[(u, tidx_a, k)] += w_a
                nlk[(tidx_a, k)] += w_a
            nuk[(u, k)] += 1
            nk[k] += 1

        yield i, ntk, ndk, nulk


def read_one_record(filename):
    with open(filename, 'r') as fin:
        for uid_str, time_str, tower_str, lat_str, lon_str in csv.reader(fin, delimiter='\t'):

            if lat_str == 'NULL' or lon_str == 'NULL':
                continue

            uid = uuid.UUID(uid_str)

            tstamp = time.mktime(time.strptime(time_str, '%Y-%m-%d %H:%M:%S'))
            init_t = time.mktime(time.strptime('2013-08-12 00:00:00', '%Y-%m-%d %H:%M:%S'))
            timeslot = int(tstamp - init_t) / 3600
            t = timeslot % 24
            d = timeslot / 24

            tid = int(tower_str, 16)

            yield uid, t, d, tid


def read_records(filename, tower_index):
    user_logs = []
    uidx_p = 0
    for uid, t, d, tid in read_one_record(filename):
        if uid not in users:
            users[uid] = uidx_p
            uidx_p += 1
        uidx = users[uid]

        user_logs.append((uidx, t, d, tower_index[tid][0]))

    return user_logs


def main():
    tower_index, tower_idx2id = read_tower_index('./tower_coordinates.csv')
    tower_graph = np.loadtxt('./tower_dis_graph_simplified.csv', delimiter=',')
    affinity_sparse_matrix = get_sparse_affinity(tower_index, tower_idx2id, tower_graph)
    del tower_graph
    # print affinity_sparse_matrix
    # for tidx1 in affinity_sparse_matrix:
        # for tidx2, w in affinity_sparse_matrix[tidx1]:
            # print '{}, {}, {}'.format(tidx1, tidx2, w)

    user_logs = read_records('./5000traj.csv', tower_index)
    print 'Read file complete'

    for i, ntk, ndk, nulk in crowdtopic(user_logs, tower_index, affinity_sparse_matrix):

        np.savetxt('./rst/ntk_iter{}.csv'.format(i), ntk, delimiter=',')
        np.savetxt('./rst/ndk_iter{}.csv'.format(i), ndk, delimiter=',')
        for k in xrange(K):
            np.savetxt('./rst/nulk_iter{}_k{}.csv'.format(i, k), nulk[:, :, k], delimiter=',')

if __name__ == '__main__':
    main()
