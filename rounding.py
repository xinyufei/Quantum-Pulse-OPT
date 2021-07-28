from pycombina import *
import numpy as np
import matplotlib.pyplot as plt
from qutip import Qobj, identity, sigmax, sigmaz, sigmay, tensor
from qutip.qip.operations.gates import cnot
from evolution import *


def rounding(file_name, time_interval, num_time_step, type, min_up_times=0, b_rel=None):
    file = open(file_name)
    # b_rel_origin = np.loadtxt(file, delimiter=",")
    if b_rel is not None:
        b_rel_origin = b_rel
    else:
        b_rel_origin = np.loadtxt(file, delimiter=",")
    # b_rel_origin = np.loadtxt(file)
    # t = np.array([t_step for t_step in range(b_rel.shape[0] + 1)])
    t = np.linspace(0, time_interval, num_time_step + 1)
    b_rel = np.zeros((num_time_step, b_rel_origin.shape[1]))
    step = num_time_step / b_rel_origin.shape[0]
    for j in range(b_rel_origin.shape[1]):
        for time_step in range(num_time_step):
            b_rel[time_step, j] = b_rel_origin[int(np.floor(time_step / step)), j]
    # ts_interval = 20

    binapprox = BinApprox(t, b_rel)

    if type == "SUR":
        min_up_times = "SUR"
        sur = CombinaSUR(binapprox)
        sur.solve()

    if type == "BnB":
        # binapprox.set_n_max_switches(n_max_switches=[min_up_times, min_up_times])
        binapprox.set_min_up_times(min_up_times=[min_up_times] * b_rel.shape[1])
        combina = CombinaBnB(binapprox)
        combina.solve()

    b_bin = binapprox.b_bin

    for control_idx in range(int(np.ceil(b_bin.shape[0] / 2))):
        f, (ax1, ax2) = plt.subplots(2, sharex=True)
        ax1.step(t[:-1], b_rel[:, control_idx * 2], label="b_rel", color="C0", linestyle="dashed", where="post")
        ax1.step(t[:-1], b_bin[control_idx * 2, :], label="b_bin", color="C0", where="post")
        ax1.legend(loc="upper left")
        ax1.set_ylabel("u_" + str(control_idx * 2 + 1))
        if control_idx * 2 + 1 < b_bin.shape[0]:
            ax2.step(t[:-1], b_rel[:, control_idx * 2 + 1], label="b_rel", color="C1", linestyle="dashed", where="post")
            ax2.step(t[:-1], b_bin[control_idx * 2 + 1, :], label="b_bin", color="C1", where="post")
            ax2.legend(loc="upper left")
            ax2.set_ylabel("u_" + str(control_idx * 2 + 2))
        plt.savefig(file_name.split(".csv")[0] + "_bvsr_" + str(min_up_times) + "_" + str(control_idx * 2 + 1) + "+" +
                    str(control_idx * 2 + 2) + ".png")

    fig = plt.figure(dpi=300)
    plt.title("Rounded Optimised Control Sequences")
    plt.xlabel("Time")
    plt.ylabel("Control amplitude")
    for j in range(b_rel.shape[1]):
        plt.step(t, np.hstack((b_bin[j, :], b_bin[j, -1])),
                 where='post', linewidth=2)
    plt.savefig(file_name.split(".csv")[0] + "_binary_" + str(min_up_times) + ".png")

    return b_rel, b_bin


if __name__ == '__main__':
    file_name = "control/CNOTSUM1_evotime10_n_ts200_ptypeZERO_offset0.5_objUNIT.csv"

    alpha = 0.01
    # file_name = "control-tr/CNOTSUM1_evotime10_n_ts200_ptypeZERO_offset0.5_objUNIT_binary_0.5_sigma0.25_eta0.001_threshold30 3.csv"
    # file_name = "control-tr/n_ts200_n_ctrl2_alpha0.001_sigma0.25_eta0.001_0.1_random_7.tsv"
    evo_time = 10
    time_step = evo_time * 20
    b_rel = None
    b_rel = np.zeros((time_step, 2))
    t = 0
    for line in open(file_name, "r"):
        print(line.split(" "))
        if line.split(" "):
            for j in range(2):
                b_rel[t, j] = float(line.split(",")[j])
            t += 1

    fig = plt.figure(dpi=300)
    plt.title("Optimised Quantum Control Sequences")
    plt.xlabel("Time")
    plt.ylabel("Control amplitude")
    plt.ylim([0, 1])
    for j in range(b_rel.shape[1]):
        plt.step(np.linspace(0, 10, 201), np.hstack((b_rel[:, j], b_rel[-1, j])),
                 where='post', linewidth=2)
    # plt.axis('off')
    plt.savefig(file_name.split(".csv")[0] + "_continuous" + ".png")

    b_rel, b_bin = rounding(file_name, evo_time, time_step, "SUR", 0, b_rel = b_rel)
    np.savetxt(file_name.split(".csv")[0] + "_binary_" + str(0) + ".csv", b_bin.T)
    # b_rel, b_bin = rounding(file_name, 10, 200, "SUR")
    # The control Hamiltonians (Qobj classes)
    H_c = [tensor(sigmax(), identity(2)), tensor(sigmay(), identity(2))]
    # Drift Hamiltonian
    H_d = tensor(sigmax(), sigmax()) + tensor(sigmay(), sigmay()) + tensor(sigmaz(), sigmaz())
    # start point for the gate evolution
    X_0 = identity(4)
    # Target for the gate evolution
    X_targ = cnot()
    rounding_result = time_evolution(
        H_d.full(), [h_c.full() for h_c in H_c], time_step, evo_time, b_bin.T, X_0.full(), False, 1)
    output = open(file_name.split(".csv")[0] + "_binary_" + str(0) + ".log", "a+")
    print("obj", compute_obj(X_targ, rounding_result), file=output)
    print("obj-norm", compute_obj_with_TV(X_targ, rounding_result, b_bin.T, len(H_c), alpha), file=output)
    print("norm", sum(sum(abs(b_bin.T[t, j] - b_bin.T[t + 1, j]) for j in range(2))for t in range(time_step - 1)), file=output)