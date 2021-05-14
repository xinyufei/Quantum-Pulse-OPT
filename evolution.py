import numpy as np
from scipy.linalg import expm


def time_evolution(H_d, H_c, n_ts, evo_time, u_list, X_0, sum_cons_1):
    H_origin_c = [h_c.copy() for h_c in H_c]
    if sum_cons_1:
        H_d_new = H_d + H_origin_c[-1]
        H_d = H_d_new.copy()
        H_c = [(H_origin_c[i] - H_origin_c[-1]).copy() for i in range(len(H_origin_c) - 1)]

    n_ctrls = len(H_c)
    delta_t = evo_time / n_ts
    X = [X_0]
    for t in range(n_ts):
        H_t = H_d.copy()
        for j in range(n_ctrls):
            H_t += u_list[t, j] * H_c[j].copy()
        X_t = expm(-1j*H_t*delta_t).dot(X[t])
        X.append(X_t)
    return X[-1]


def compute_obj(U_targ, U_result):
    fid = np.abs(np.trace((np.linalg.inv(U_targ.full()).dot(U_result)))) / U_targ.full().shape[0]
    obj = 1 - fid
    return obj
