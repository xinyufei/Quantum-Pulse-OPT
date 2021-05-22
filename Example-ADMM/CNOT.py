import os
import numpy as np
from qutip import Qobj, identity, sigmax, sigmaz, sigmay, tensor
from qutip.qip.operations.gates import cnot

import sys
sys.path.append("..")
from rounding import rounding
from evolution import time_evolution, compute_obj, compute_obj_with_TV
from optcontrol_admm import Optcontrol_ADMM


if_rounding = True
if_warm_start = False
# if_rounding = True

sum_cons_1 = True

# QuTiP control modules
# a two-qubit system with target control mode as CNOT gate
example_name = 'CNOT'

# with summation one constraint
if sum_cons_1:
    example_name = 'CNOTSUM1FPRIME2'

# The control Hamiltonians (Qobj classes)
H_c = [tensor(sigmax(), identity(2)), tensor(sigmay(), identity(2))]
# Drift Hamiltonian
H_d = tensor(sigmax(), sigmax()) + tensor(sigmay(), sigmay()) + tensor(sigmaz(), sigmaz())
# start point for the gate evolution
X_0 = identity(4)
# Target for the gate evolution
X_targ = cnot()

# Time allowed for the evolution
evo_time = 1
# Number of time slots
n_ts = 20 * evo_time

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
offset = 0.5
# p_type = "RND"
# offset = 0
# objective value type
obj_type = "UNIT"
# file of the initial control for the warm start
initial_control = None  # no warm start

# ADMM parameter
alpha = 0.05
rho = 0.25
max_iter_admm = 200
max_wall_time_admm = 7200 * 100
admm_err_targ = 1e-5

os.chdir(sys.path[0])

if if_warm_start:
    # warm start
    p_type = "WARM"
    offset = 0
    initial_control = "../control-ADMM/" + "{}_evotime{}_n_ts{}_ptype{}_offset{}_obj{}_penalty{}_ADMM_{}_iter{}".format(
        example_name, 10, 200, "ZERO", 0.5, obj_type, alpha, rho, max_iter_admm) + ".csv"

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
CNOT_opt_admm = Optcontrol_ADMM()
CNOT_opt_admm.build_optimizer(H_d.full(), [h_c.full() for h_c in H_c], X_0.full(), X_targ.full(), n_ts, evo_time,
                              amp_lbound=0, amp_ubound=1,
                              fid_err_targ=fid_err_targ, min_grad=min_grad, max_iter_step=max_iter_step,
                              max_wall_time_step=max_wall_time_step, fid_type="UNIT", phase_option="PSU",
                              p_type=p_type, seed=None, constant=offset, initial_control=initial_control,
                              output_num=output_num, output_fig=output_fig, output_control=output_control,
                              sum_cons_1=sum_cons_1,
                              alpha=alpha, rho=rho, max_iter_admm=max_iter_admm, max_wall_time_admm=max_wall_time_admm,
                              admm_err_targ=admm_err_targ)
CNOT_opt_admm.optimize_admm()


b_rel = np.loadtxt(output_control, delimiter=",")
c_result = time_evolution(
        H_d.full(), [h_c.full() for h_c in H_c], n_ts, evo_time, b_rel, X_0.full(), sum_cons_1)
f = open(output_num, "a+")
print("Final objective value with penalty: ", file=f)
print(compute_obj_with_TV(X_targ, c_result, b_rel, 1, alpha), file=f)
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
        H_d.full(), [h_c.full() for h_c in H_c], n_ts, evo_time, bin_amps.T, X_0.full(), sum_cons_1)
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

