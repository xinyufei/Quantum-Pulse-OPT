import os
import numpy as np
from qutip import Qobj, identity, sigmax, sigmaz, sigmay, tensor
import matplotlib.pyplot as plt
import sys

sys.path.append("..")
# sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])
from optcontrol import optcontrol
from optcontrol_penalized import Optcontrol_Penalized
from rounding import rounding
from evolution import time_evolution, compute_obj, compute_obj_with_TV, compute_sum_cons
from auxiliary_function import *

qubit_num = 5

# Choose optimizing State transfer or Unitary gate
state_transfer = False

# Choose whether include intermediate state evolution as part of the graph optimization
use_inter_vecs = False

# Defining H0
qubit_state_num = 2
freq_ge = 0  # GHz
g = 2 * np.pi * 0.1  # GHz

Q_x = np.diag(np.sqrt(np.arange(1, qubit_state_num)), 1) + np.diag(np.sqrt(np.arange(1, qubit_state_num)), -1)
Q_y = (0 + 1j) * (
            np.diag(np.sqrt(np.arange(1, qubit_state_num)), 1) - np.diag(np.sqrt(np.arange(1, qubit_state_num)), -1))
sigma_z = np.diag(np.arange(-1, qubit_state_num, 2))
I_q = np.identity(qubit_state_num)

g_op = nn_chain_kron(sigma_z, I_q, qubit_num, qubit_state_num)

H0 = g * g_op

# Defining dressed info
is_dressed = False
#     w_c, v_c, dressed_id = get_dressed_info(H0)
#     dressed_info = {'dressed_id':dressed_id, 'eigenvectors':v_c, 'eigenvalues':w_c,'is_dressed':is_dressed}

dressed_info = None

# Defining Concerned states (starting states)

psi0 = concerned(qubit_num, qubit_state_num)

# Defining states to include in the drawing of occupation
states_draw_list = range(qubit_state_num ** qubit_num)
states_draw_names = []
for ii in states_draw_list:
    states_draw_names.append(Basis(ii, qubit_num, qubit_state_num))

# Defining U (Target)

if is_dressed:
    U = dressed_unitary(Hadamard(qubit_num), v_c, dressed_id)
else:
    U = Hadamard(qubit_num)

# print U

if state_transfer:
    target_vec_list = []

    for ii in psi0:
        target_vec = np.dot(U, v_c[:, get_state_index(ii, dressed_id)])
        target_vec_list.append(target_vec)

    U = target_vec_list

# Defining U0 (Initial)
q_identity = np.identity(qubit_state_num ** qubit_num)
U0 = q_identity

# Defining control Hs

Hops = []
Hnames = []
ops_max_amp = []
max_amp = 2 * np.pi * 0.5
Hops, Hnames, ops_max_amp = append_separate_krons(Q_x, 'x', qubit_num, qubit_state_num, Hops, Hnames, ops_max_amp,
                                                  amp=max_amp)
Hops, Hnames, ops_max_amp = append_separate_krons(Q_y, 'y', qubit_num, qubit_state_num, Hops, Hnames, ops_max_amp,
                                                  amp=max_amp)
# Hops,Hnames,ops_max_amp = append_separate_krons(Q_y,'z',qubit_num,qubit_state_num,Hops,Hnames,ops_max_amp,amp=max_amp)

# print Hops[0]
# print(ops_max_amp)
if_rounding = True
if_warm_start = False
# if_rounding = True

sum_cons_1 = False

# QuTiP control modules
# a two-qubit system with target control mode as CNOT gate
example_name = 'SPIN' + str(qubit_num)

# with summation one constraint
if sum_cons_1:
    example_name = 'SPINSUM1' + str(qubit_num)

# The control Hamiltonians (Qobj classes)
H_c = [Qobj(hops) for hops in Hops]
# Drift Hamiltonian
H_d = Qobj(H0)
# start point for the gate evolution
X_0 = Qobj(U0)
# Target for the gate evolution
X_targ = Qobj(U)

# Defining time scales
# total_time = qubit_num * 2.0
# steps = qubit_num * 10
# Time allowed for the evolution
evo_time = qubit_num * 4
# Number of time steps
n_ts = 10 * evo_time * 1

# Fidelity error target
fid_err_targ = 1e-4
# Maximum iterations for the optimise algorithm
max_iter = 10000
# Maximum (elapsed) time allowed in seconds
max_wall_time = 7200 * 4
max_iter_step = 500
# Minimum gradient (sum of gradients squared)
# as this tends to 0 -> local minimum has been found
min_grad = 1e-6

# initialized type
p_type = "ZERO"
offset = 0.5
# objective value type
obj_type = "UNIT"
# file of the initial control for the warm start
initial_control = None  # no warm start

sum_penalty = 100
max_controllers = 1

os.chdir(sys.path[0])

if if_warm_start:
    # warm start
    p_type = "WARM"
    offset = 0
    initial_control = "../control/" + "{}_evotime{}_n_ts{}_ptype{}_offset{}_obj{}".format(
        "SPINWARM" + str(qubit_num), evo_time, n_ts, "WARM", 0, obj_type, 0, sum_penalty) + ".csv"

if not os.path.exists("../output-SUR/"):
    os.makedirs("../output-SUR/")
if not os.path.exists("../control-SUR/"):
    os.makedirs("../control-SUR/")

alpha = 0

p_type = "WARM"
offset = 0

output_num = "../output-SUR/" + "{}_evotime{}_n_ts{}_ptype{}_offset{}_obj{}_penalty{}_sum_penalty{}".format(
    example_name, evo_time, n_ts, p_type, offset, obj_type, alpha, sum_penalty) + ".log"
output_fig = "../output-SUR/" + "{}_evotime{}_n_ts{}_ptype{}_offset{}_obj{}_penalty{}_sum_penalty{}".format(
    example_name, evo_time, n_ts, p_type, offset, obj_type, alpha, sum_penalty) + ".png"
output_control = "../control/" + "{}_evotime{}_n_ts{}_ptype{}_offset{}_obj{}_penalty{}_sum_penalty{}".format(
    example_name, evo_time, n_ts, p_type, offset, obj_type, alpha, sum_penalty) + ".csv"

output_control = "../control-tr/SPIN5_evotime20_n_ts200_ptypeZERO_offset0.5_objUNIT_penalty0_sum_penalty0_binary_SUR_alpha0.01_sigma0.25_eta0.001_threshold30.csv"
# solve the optimization model
# import cProfile
#
# cProfile.run(
# "optcontrol(example_name, H_d, H_c, X_0, X_targ, n_ts, evo_time, p_type, initial_control, output_num, output_fig, output_control, sum_cons_1, fid_err_targ, max_iter, max_wall_time, min_grad, offset)", filename="profiler_result.out")
# Hadamard_penalized = Optcontrol_Penalized()
# Hadamard_penalized.build_optimizer(H_d.full(), [h_c.full() for h_c in H_c], X_0.full(), X_targ.full(), n_ts, evo_time,
#                               amp_lbound=0, amp_ubound=1, ops_max_amp=ops_max_amp,
#                               fid_err_targ=fid_err_targ, min_grad=min_grad,
#                               max_iter_step=max_iter_step, fid_type="UNIT", phase_option="PSU",
#                               p_type=p_type, seed=None, constant=offset, initial_control=initial_control,
#                               output_num=output_num, output_fig=output_fig, output_control=output_control,
#                               penalty=sum_penalty, max_controllers=1)
#
# import cProfile
# cProfile.run("Hadamard_penalized.optimize_penalized()", filename="profiler_result.out")
# Hadamard_penalized.optimize_penalized()


# b_rel = np.loadtxt(output_control, delimiter=" ")
b_rel = np.zeros((n_ts, len(H_c)))
t = 0
for line in open(output_control, "r"):
    print(line.split(" "))
    if line.split(" "):
        for j in range(len(H_c)):
            b_rel[t, j] = float(line.split(" ")[j])
        t += 1


# fig = plt.figure(dpi=300)
# plt.title("Optimised Quantum Control Sequences")
# plt.xlabel("Time")
# plt.ylabel("Control amplitude")
# plt.ylim([0,1])
# for j in range(b_rel.shape[1]):
#     plt.step(np.linspace(0, 8, 81), np.hstack((b_rel[:, j], b_rel[-1, j])),
#              where='post', linewidth=2)
# # plt.axis('off')
# plt.savefig(output_control.split(".csv")[0] + "_continuous" + ".png")

# output the figures
# time_list = np.array([t * self.evo_time / self.n_ts for t in range(self.n_ts + 1)])
# fig1 = plt.figure()
# ax1 = fig1.add_subplot(2, 1, 1)
# ax1.set_title("Initial control amps")
# # ax1.set_xlabel("Time")
# ax1.set_ylabel("Control amplitude")
# for j in range(self.n_ctrls):
#     ax1.step(time_list, np.hstack((initial_amps[:, j], initial_amps[-1, j])), where='post')
#
# ax2 = fig1.add_subplot(2, 1, 2)
# ax2.set_title("Optimised Control Sequences")
# ax2.set_xlabel("Time")
# ax2.set_ylabel("Control amplitude")
# for j in range(final_amps.shape[1]):
#     ax2.step(time_list, np.hstack((final_amps[:, j], final_amps[-1, j])), where='post')
# # if self.sum_cons_1:
# #     ax2.step(np.array([t for t in range(self.n_ts)]),
# #              np.hstack((final_amps[:, self.n_ctrls], final_amps[-1, self.n_ctrls])), where='post')
# plt.tight_layout()
# if self.output_fig:
#     plt.savefig(self.output_fig)

# print(sum([(sum(b_rel[t, j] for j in range(len(H_c))) - 1) * evo_time/n_ts for t in range(n_ts)]) / 8)
# c_result = time_evolution(
#         H_d.full(), [h_c.full() for h_c in H_c], n_ts, evo_time, b_rel, X_0.full(), False, ops_max_amp)
# f = open(output_num, "a+")
# print("Final objective value with norm: ", file=f)
# print(compute_obj_with_TV(X_targ, c_result, b_rel, 1, alpha), file=f)
# print("Final penalty of max controllers: ", file=f)
# print(compute_sum_cons(b_rel, max_controllers), file=f)
# f.close()

if if_rounding:
    rounding_type = "BnB"
    min_up_time = 1

    if rounding_type == "SUR":
        min_up_time = "SUR"


    rounding_ts = n_ts

    # do the rounding
    b_rel_r, bin_amps = rounding(output_control, evo_time, rounding_ts, rounding_type, min_up_time, b_rel=b_rel)

    # exit()
    sum_norm = np.zeros((1, rounding_ts))
    integral = np.zeros((len(H_c), rounding_ts + 1))
    delta_t = evo_time / rounding_ts
    for t_r in range(rounding_ts):
        for j in range(len(H_c)):
            integral[j, t_r + 1] = \
                integral[j, t_r] + (b_rel_r[t_r, j] - bin_amps[j, t_r]) * delta_t
        sum_norm[0, t_r] = sum(integral[j, t_r + 1] for j in range(len(H_c)))
    print(np.max(sum_norm), np.min(sum_norm))

    # exit()
    # evolution results by the control list after the rounding
    rounding_result = time_evolution(
        H_d.full(), [h_c.full() for h_c in H_c], rounding_ts, evo_time, bin_amps.T, X_0.full(), False, ops_max_amp)
    np.savetxt(output_control.split(".csv")[0] + "_binary_" + str(min_up_time) + ".csv", bin_amps.T)
    f = open(output_control.split(".csv")[0] + "_binary_" + str(min_up_time) + ".log", "w+")
    print("Rounding result: ", file=f)
    print(rounding_result, file=f)
    print("Final objective value: ", file=f)
    print(compute_obj(X_targ, rounding_result), file=f)
    print("Final objective value with norm: ", file=f)
    print(compute_obj_with_TV(X_targ, rounding_result, bin_amps.T, 1, alpha), file=f)
    print("Final penalty of max controllers: ", file=f)
    print(compute_sum_cons(bin_amps.T, max_controllers), file=f)
    print("Infinity norm of continuous and binary controllers", file=f)
    inf_norm = np.zeros((len(H_c), rounding_ts))
    integral = np.zeros((len(H_c), rounding_ts + 1))
    delta_t = evo_time / rounding_ts
    for j in range(len(H_c)):
        for t_r in range(rounding_ts):
            integral[j, t_r + 1] = \
                integral[j, t_r] + (b_rel_r[t_r, j] - bin_amps[j, t_r]) * delta_t
            inf_norm[j, t_r] = abs(integral[j, t_r + 1])
    print(np.max(np.max(inf_norm)), file=f)
    f.close()

