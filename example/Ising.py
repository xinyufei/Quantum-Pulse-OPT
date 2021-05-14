import os
from qutip import Qobj, identity, sigmax, sigmaz
from code_Brady_opt.generate_matrix import *

from optcontrol import optcontrol
from rounding import rounding
from evolution import time_evolution, compute_obj

if_rounding = False
# if_rounding = True
if rounding:
    rounding_type = "BnB"
    min_up_time = 10
if_warm_start = False


# QuTiP control modules
example_name = 'Ising'
# set the dimension
n = 4
# Generate J_ij
# generate_Jij_MC(n, 3)
generate_Jij(n)
# generate_Jij_LR(n, 1, 2)
# Generate Hamiltonian
C_mat = get_ham(n, True)
B_mat = get_ham(n, False)
# The (single) control Hamiltonian
H_c = [Qobj(B_mat - C_mat)]
# Drift Hamiltonian
H_d = Qobj(C_mat)
# start point for the gate evolution
Ground_B = Qobj(B_mat).groundstate()[1]
X_0 = Ground_B
# Target for the gate evolution
Ground_C = Qobj(C_mat).groundstate()[1]
X_targ = Ground_C

# Number of time slots
n_ts = 16
# Time allowed for the evolution
evo_time = 1

# Fidelity error target
fid_err_targ = 1e-10
# Maximum iterations for the optisation algorithm
max_iter = 500
# Maximum (elapsed) time allowed in seconds
max_wall_time = 120
# Minimum gradient (sum of gradients squared)
# as this tends to 0 -> local minima has been found
min_grad = 1e-50

p_type = "ZERO"
offset = 0.5
obj_type = "UNIT"
initial_control = None

if if_warm_start:
    # warm start
    p_type = "WARM"
    offset = 0
    initial_control = "../control/" + "{}_evotime{}_n_ts{}_ptype{}_offset{}_obj{}".format(
        example_name, 1, 16, "ZERO", 0.5, obj_type) + ".csv"

if not os.path.exists("../output/"):
    os.makedirs("../output/")
if not os.path.exists("../control/"):
    os.makedirs("../control/")

output_num = "../output/" + "{}_evotime{}_n_ts{}_ptype{}_offset{}_obj{}".format(
    example_name, evo_time, n_ts, p_type, offset, obj_type) + ".log"
output_fig = "../output/" + "{}_evotime{}_n_ts{}_ptype{}_offset{}_obj{}".format(
    example_name, evo_time, n_ts, p_type, offset, obj_type) + ".png"
output_control = "../control/" + "{}_evotime{}_n_ts{}_ptype{}_offset{}_obj{}".format(
    example_name, evo_time, n_ts, p_type, offset, obj_type) + ".csv"

optcontrol(example_name, H_d, H_c, X_0, X_targ, n_ts, evo_time, p_type, initial_control, output_num, output_fig, output_control,
           False, fid_err_targ, max_iter, max_wall_time, min_grad, offset)

if if_rounding:

    bin_amps = rounding(output_control, rounding_type, min_up_time)

    rounding_result = time_evolution(
        H_d.full(), [h_c.full() for h_c in H_c], evo_time, n_ts, bin_amps.T, X_0.full())
    f = open(output_control.split(".csv")[0] + "_binary_" + str(min_up_time) + ".log", "w+")
    print("Rounding result: ", file=f)
    print(rounding_result, file=f)
    print("Final objective value: ", file=f)
    print(compute_obj(X_targ, rounding_result), file=f)

