import numpy as np
from scipy.linalg import expm


def time_evolution(H_d, H_c, n_ts, evo_time, u_list, X_0, sum_cons_1, ops_max_amp):
    if not isinstance(ops_max_amp, list):
        max_amp = [ops_max_amp] * len(H_c)
    else:
        max_amp = ops_max_amp
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
            H_t += u_list[t, j] * max_amp[j] * H_c[j].copy()
        X_t = expm(-1j*H_t*delta_t).dot(X[t])
        X.append(X_t)
    return X[-1]


def compute_obj(U_targ, U_result, obj_type="UNIT"):
    if obj_type == "UNIT":
        fid = np.abs(np.trace((np.linalg.inv(U_targ.full()).dot(U_result)))) / U_targ.full().shape[0]
    if obj_type == "GEN_SQUARE":
        fid = np.square(np.abs(np.trace(U_targ.full().conj().T.dot(U_result))))
    obj = 1 - fid
    return obj


def compute_tv_norm(u_list):
    if len(u_list.shape) == 1:
        return sum(abs(u_list[time_step + 1] - u_list[time_step]) for time_step in range(u_list.shape[0] - 1))
    else:
        n_ctrls = u_list.shape[1]
        return sum(sum(abs(u_list[time_step + 1, j] - u_list[time_step, j]) for time_step in range(u_list.shape[0] - 1))
                   for j in range(n_ctrls))


def compute_obj_with_TV(U_targ, U_result, u_list, n_ctrls, alpha, obj_type="UNIT"):
    obj = compute_obj(U_targ, U_result, obj_type)
    TV = compute_tv_norm(u_list)
    return obj + alpha * TV


def compute_sum_cons(u_list, max_controllers):
    n_ctrls = u_list.shape[1]
    n_ts = u_list.shape[0]
    penalty = sum(np.power(sum(u_list[t, j] for j in range(n_ctrls)) - max_controllers, 2) for t in range(n_ts))
    return penalty