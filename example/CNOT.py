import os
import numpy as np
from qutip import Qobj, identity, sigmax, sigmaz, sigmay, tensor
from qutip.qip.operations.gates import cnot

import sys
sys.path.append("..")
from optcontrol import optcontrol
from rounding import rounding
from evolution import time_evolution, compute_obj, compute_obj_with_TV


if_rounding = True
if_warm_start = False
# if_rounding = True

sum_cons_1 = True

# QuTiP control modules
# a two-qubit system with target control mode as CNOT gate
example_name = 'CNOT'

# with summation one constraint
if sum_cons_1:
    example_name = 'CNOTSUM1FPRIME'

# The control Hamiltonians (Qobj classes)
H_c = [tensor(sigmax(), identity(2)), tensor(sigmay(), identity(2))]
# Drift Hamiltonian
H_d = tensor(sigmax(), sigmax()) + tensor(sigmay(), sigmay()) + tensor(sigmaz(), sigmaz())
# start point for the gate evolution
X_0 = identity(4)
# Target for the gate evolution
X_targ = cnot()

# Time allowed for the evolution
evo_time = 20
# Number of time steps
n_ts = 20 * evo_time

# Fidelity error target
fid_err_targ = 1e-8
# Maximum iterations for the optimise algorithm
max_iter = 500
# Maximum (elapsed) time allowed in seconds
max_wall_time = 120
# Minimum gradient (sum of gradients squared)
# as this tends to 0 -> local minimum has been found
min_grad = 1e-8

# initialized type
p_type = "ZERO"
offset = 0.5
# objective value type
obj_type = "UNIT"
# file of the initial control for the warm start
initial_control = None  # no warm start

os.chdir(sys.path[0])

if if_warm_start:
    # warm start
    p_type = "WARM"
    offset = 0
    initial_control = "../control/" + "{}_evotime{}_n_ts{}_ptype{}_offset{}_obj{}".format(
        example_name, 10, 200, "ZERO", 0.5, obj_type) + ".csv"

if not os.path.exists("../output/"):
    os.makedirs("../output/")
if not os.path.exists("../control/"):
    os.makedirs("../control/")

alpha = 0.1

output_num = "../output/" + "{}_evotime{}_n_ts{}_ptype{}_offset{}_obj{}_penalty{}".format(
    example_name, evo_time, n_ts, p_type, offset, obj_type, alpha) + ".log"
output_fig = "../output/" + "{}_evotime{}_n_ts{}_ptype{}_offset{}_obj{}_penalty{}".format(
    example_name, evo_time, n_ts, p_type, offset, obj_type, alpha) + ".png"
output_control = "../control/" + "{}_evotime{}_n_ts{}_ptype{}_offset{}_obj{}_penalty{}".format(
    example_name, evo_time, n_ts, p_type, offset, obj_type, alpha) + ".csv"

# solve the optimization model
optcontrol(example_name, H_d, H_c, X_0, X_targ, n_ts, evo_time, p_type, initial_control, output_num, output_fig,
           output_control, sum_cons_1, fid_err_targ, max_iter, max_wall_time, min_grad, offset)


b_rel = np.loadtxt(output_control, delimiter=",")
c_result = time_evolution(
        H_d.full(), [h_c.full() for h_c in H_c], n_ts, evo_time, b_rel, X_0.full(), sum_cons_1)
f = open(output_num, "a+")
print("Final objective value with norm: ", file=f)
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
    print(compute_obj_with_TV(X_targ, rounding_result, bin_amps.T, 1, alpha), file=f)
    f.close()

