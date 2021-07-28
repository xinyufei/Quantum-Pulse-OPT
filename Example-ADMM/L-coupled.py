import os
import numpy as np
from qutip import Qobj, identity, sigmax, sigmaz, sigmay, tensor
from qutip.qip.operations.gates import cnot

import sys
sys.path.append("..")
# from rounding import rounding
from evolution import time_evolution, compute_obj, compute_obj_with_TV, compute_tv_norm
from optcontrol_admm import Optcontrol_ADMM


if_rounding = False
# if_warm_start = True
if_warm_start = False
# if_rounding = True

sum_cons_1 = False

num_qubits = 1

# QuTiP control modules
# a two-qubit system with target control mode as CNOT gate
example_name = 'Lcouple' + str(num_qubits)

# with summation one constraint
if sum_cons_1:
    example_name = 'LcoupleSUM1' + str(num_qubits)

# generate sigma matrix for each qubit
X_list = []  # list of Qobjs
Y_list = []
Z_list = []
for i in range(num_qubits):
    if i == 0:
        cur_tensor_x = sigmax()
        cur_tensor_y = sigmay()
        cur_tensor_z = sigmaz()
    else:
        cur_tensor_x = identity(2)
        cur_tensor_y = identity(2)
        cur_tensor_z = identity(2)
    for j in range(1, num_qubits):
        if j == i:
            cur_tensor_x = tensor(cur_tensor_x, sigmax())
            cur_tensor_y = tensor(cur_tensor_y, sigmay())
            cur_tensor_z = tensor(cur_tensor_z, sigmaz())
        else:
            cur_tensor_x = tensor(cur_tensor_x, identity(2))
            cur_tensor_y = tensor(cur_tensor_y, identity(2))
            cur_tensor_z = tensor(cur_tensor_z, identity(2))
    X_list.append(cur_tensor_x)
    Y_list.append(cur_tensor_y)
    Z_list.append(cur_tensor_z)

# generate the drift Hamiltonian
drift_list = []  # list of matrices
for i in range(num_qubits - 1):
    drift_list.append(Z_list[i + 1].full().dot(Z_list[i].full()))
H_d = -sum(drift_list) - sum(Z_list)

# generate the control Hamiltonian
H_c = [-sum(X_list)]

# start point for the gate evolution
start_mat = H_d - 2 * H_c[0]
X_0 = Qobj(start_mat).groundstate()[1]

print(X_0)

# Target for the gate evolution
target_mat = H_d + 2 * H_c[0]
X_targ = Qobj(target_mat).groundstate()[1]

print(X_targ)

# Time allowed for the evolution
evo_time = 1.5
# Number of time slots
n_ts = int(20 * evo_time)

# Fidelity error target
fid_err_targ = 1e-8
# Maximum iterations for the optimise algorithm
max_iter_step = 500
# Maximum (elapsed) time allowed in seconds
max_wall_time_step = 240
# Minimum gradient (sum of gradients squared)
# as this tends to 0 -> local minimum has been found
min_grad = 1e-8

# initialized type
p_type = "ZERO"
offset = 4
# p_type = "RND"
# offset = 0
# objective value type
obj_type = "GEN_SQUARE"
# file of the initial control for the warm start
initial_control = None  # no warm start

# ADMM parameter
alpha = 0
rho = 0
max_iter_admm = 1
max_wall_time_admm = 7200 * 100
admm_err_targ = 1e-6

os.chdir(sys.path[0])

if if_warm_start:
    # warm start
    p_type = "WARM"
    offset = 0
    # initial_control = "../control-ADMM/" + "{}_evotime{}_n_ts{}_ptype{}_offset{}_obj{}_penalty{}_ADMM_{}_iter{}".format(
    #     example_name, evo_time, int(evo_time * 20), "ZERO", 0, obj_type, 0, 0, 1) + ".csv"
    initial_control = "../control-ADMM/" + "{}_evotime{}_n_ts{}_ptype{}_offset{}_obj{}_penalty{}_ADMM_{}_iter{}".format(
        example_name, evo_time, int(evo_time * 20), "WARM", 0, obj_type, 1e-1, 0.5, max_iter_admm) + ".csv"

if not os.path.exists("../output-ADMM/"):
    os.makedirs("../output-ADMM/")
if not os.path.exists("../control-ADMM/"):
    os.makedirs("../control-ADMM/")

output_num = "../output-ADMM/" + "{}_evotime{}_n_ts{}_ptype{}_offset{}_obj{}_penalty{}_ADMM_{}_iter{}".format(
        example_name, evo_time, n_ts, p_type, offset, obj_type, alpha, rho, max_iter_admm) + ".log"
output_fig = "../output-ADMM/" + "{}_evotime{}_n_ts{}_ptype{}_offset{}_obj{}_penalty{}_ADMM_{}_iter{}".format(
        example_name, evo_time, n_ts, p_type, offset, obj_type, alpha, rho, max_iter_admm) + ".png"
output_control = "../control-ADMM/" + "{}_evotime{}_n_ts{}_ptype{}_offset{}_obj{}_penalty{}_ADMM_{}_iter{}".format(
        example_name, evo_time, n_ts, p_type, offset, obj_type, alpha, rho, max_iter_admm) + ".csv"

# solve the optimization model
opt_admm = Optcontrol_ADMM()
opt_admm.build_optimizer(H_d.full(), [h_c.full() for h_c in H_c], X_0.full(), X_targ.full(), n_ts, evo_time,
                              amp_lbound=-4, amp_ubound=4,
                              fid_err_targ=fid_err_targ, min_grad=min_grad, max_iter_step=max_iter_step,
                              max_wall_time_step=max_wall_time_step, fid_type=obj_type, phase_option="PSU",
                              p_type=p_type, seed=None, constant=offset, initial_control=initial_control,
                              output_num=output_num, output_fig=output_fig, output_control=output_control,
                              sum_cons_1=sum_cons_1,
                              alpha=alpha, rho=rho, max_iter_admm=max_iter_admm, max_wall_time_admm=max_wall_time_admm,
                              admm_err_targ=admm_err_targ)
opt_admm.optimize_admm()


b_rel = np.loadtxt(output_control, delimiter=",")

if len(b_rel.shape) == 1:
    b_rel = np.reshape(b_rel, (b_rel.shape[0], 1))

c_result = time_evolution(
        H_d.full(), [h_c.full() for h_c in H_c], n_ts, evo_time, b_rel, X_0.full(), sum_cons_1, 1)
f = open(output_num, "a+")
print("Final objective value with penalty: ", file=f)
obj = compute_obj(X_targ, c_result, obj_type)
print(obj, file=f)
print("Final objective value with penalty: ", file=f)
obj_TV = compute_obj_with_TV(X_targ, c_result, b_rel, 1, alpha, obj_type)
print(obj_TV, file=f)
print("Final TV norm without penalty", file=f)
TV_norm = compute_tv_norm(b_rel)
print(TV_norm, file=f)
f.close()

if if_rounding:
    rounding_type = "BnB"
    min_up_time = 10

    if rounding_type == "SUR":
        min_up_time = "SUR"

    # do the rounding
    bin_amps = rounding(output_control, rounding_type, min_up_time)

    # evolution results by the control list after the rounding
    rounding_result = time_evolution(
        H_d.full(), [h_c.full() for h_c in H_c], n_ts, evo_time, bin_amps.T, X_0.full(), 1, sum_cons_1)
    f = open(output_control.split(".csv")[0] + "_binary_" + str(min_up_time) + ".log", "w+")
    print("Rounding result: ", file=f)
    print(rounding_result, file=f)
    print("Final objective value: ", file=f)
    print(compute_obj(X_targ, rounding_result), file=f)
    print("Final objective value with norm: ", file=f)
    if sum_cons_1:
        print(compute_obj_with_TV(X_targ, rounding_result, bin_amps.T, len(H_c) - 1, alpha), file=f)
    else:
        print(compute_obj_with_TV(X_targ, rounding_result, bin_amps.T, len(H_c), alpha), file=f)
    f.close()

