import numpy as np
from scipy.linalg import expm
from qutip import Qobj, identity, sigmax, sigmaz, sigmay, tensor

# number of qubits
num_qubits = 1

# generate sigma matrix for each qubit in a multi-qubits system
X_list = []  # list of sigma_x matrix
Y_list = []  # list of sigma_y matrix
Z_list = []  # list of sigma_z matrix
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
drift_list = []
for i in range(num_qubits - 1):
    drift_list.append(Z_list[i + 1]*(Z_list[i]))
H_d = -sum(drift_list) - sum(Z_list)

# generate the control Hamiltonian
H_c = [-sum(X_list)]

# start point for the gate evolution
start_mat = H_d - 2 * H_c[0]
print(start_mat)
X_0 = start_mat.groundstate()[1]
print(X_0)

# Target for the gate evolution
target_mat = H_d + 2 * H_c[0]
print(target_mat)
X_targ = Qobj(target_mat).groundstate()[1]
print(X_targ)

print(np.square(np.abs(np.trace(X_0.full().conj().T.dot(X_targ.full())))))
# set the evolution time and pulse variable of variational control
tau_1 = 0.618
T = 0.618

# time-dependent Hamiltonians
pos_h = (H_d + 4 * H_c[0]).full()
neg_h = (H_d - 4 * H_c[0]).full()

# conduct the time evolution and compute the final fidelity
cur = X_0.full()
res = expm(-1j*tau_1/2*pos_h).dot(cur)
cur = res.copy()
res = expm(-1j*(T-tau_1)*H_d.full()).dot(cur)
cur = res.copy()
res = X_targ.full().conj().T.dot(expm(-1j*tau_1/2*neg_h).dot(cur))
print(np.square(np.abs(np.trace(res))))
