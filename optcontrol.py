import numpy as np
import matplotlib.pyplot as plt
import datetime

from qutip import Qobj, identity, sigmax, sigmaz
import qutip.logging_utils as logging
import qutip.control.pulseoptim as cpo
import qutip.control.pulsegen as pulsegen

from code_Brady_opt.generate_matrix import *

logger = logging.get_logger()
# Set this to None or logging.WARN for 'quiet' execution
log_level = logging.INFO
# QuTiP control modules

example_name = 'Optcontrol'
n = 4

# Generate J_ij
# generate_Jij_MC(n, 5)
generate_Jij(n)
# generate_Jij_LR(n, 1, 2)
# Generate Hamiltonian
C_mat = get_ham(n, True)
np.savetxt('output/C_mat' + str(n) + '.csv', C_mat)
B_mat = get_ham(n, False)
np.savetxt('output/B_mate' + str(n) + '.csv', B_mat)
# The (single) control Hamiltonian
H_c = [Qobj(B_mat - C_mat)]
# Drift Hamiltonian
H_d = Qobj(C_mat)
# start point for the gate evolution
# U_0 = identity(n * n)
Ground_B = Qobj(B_mat).groundstate()[1]
X_0 = Ground_B
# X_0 = Qobj(np.zeros((n*n, 1)))
# Target for the gate evolution
# U_targ = Qobj(C_mat)
Ground_C = Qobj(C_mat).groundstate()[1]
X_targ = Ground_C

print(X_0)
print(X_targ)

# Number of time slots
n_ts = 40
# Time allowed for the evolution
evo_time = 2
# Number of controlers
n_ctrls = len(H_c)

# Fidelity error target
fid_err_targ = 1e-10
# Maximum iterations for the optisation algorithm
max_iter = 500
# Maximum (elapsed) time allowed in seconds
max_wall_time = 120
# Minimum gradient (sum of gradients squared)
# as this tends to 0 -> local minima has been found
min_grad = 1e-50

# Set initial state
# pulse type alternatives: RND|ZERO|LIN|SINE|SQUARE|SAW|TRIANGLE|
p_type = 'RND'
offset = 0.5

# Set output files
f_ext = "{}_n_ts{}_ptype{}.txt".format(example_name, n_ts, p_type)

# Build the optimizer
optim = cpo.create_pulse_optimizer(H_d, H_c, X_0, X_targ, n_ts, evo_time,
                                   amp_lbound=0, amp_ubound=1,
                                   fid_err_targ=fid_err_targ, min_grad=min_grad,
                                   max_iter=max_iter, max_wall_time=max_wall_time, dyn_type='GEN_MAT',
                                   fid_type='TRACEDIFF', init_pulse_type=p_type, init_pulse_params={"offset": offset},
                                   log_level=log_level, gen_stats=True)

# Initialization
dyn = optim.dynamics
p_gen = optim.pulse_generator
p_gen = pulsegen.create_pulse_gen(p_type, dyn)

# Generate different initial pulses for each of the controls
init_amps = np.zeros([n_ts, n_ctrls])
for j in range(n_ctrls):
    p_gen.offset = offset
    init_amps[:, j] = p_gen.gen_pulse()
dyn.initialize_controls(init_amps)

# Run the optimization
result = optim.run_optimization()

# Report the results
result.stats.report()
print("Final evolution\n{}\n".format(result.evo_full_final))
print("********* Summary *****************")
print("Final fidelity error {}".format(result.fid_err))
print("Final gradient normal {}".format(result.grad_norm_final))
print("Terminated due to {}".format(result.termination_reason))
print("Number of iterations {}".format(result.num_iter))
print("Completed in {} HH:MM:SS.US".format(
    datetime.timedelta(seconds=result.wall_time)))

# Plot the results
fig1 = plt.figure()
ax1 = fig1.add_subplot(2, 1, 1)
ax1.set_title("Initial control amps")
# ax1.set_xlabel("Time")
ax1.set_ylabel("Control amplitude")
ax1.step(result.time,
         np.hstack((result.initial_amps[:, 0], result.initial_amps[-1, 0])),
         where='post')

ax2 = fig1.add_subplot(2, 1, 2)
ax2.set_title("Optimised Control Sequences")
ax2.set_xlabel("Time")
ax2.set_ylabel("Control amplitude")
ax2.step(result.time,
         np.hstack((result.final_amps[:, 0], result.final_amps[-1, 0])),
         where='post')
plt.tight_layout()
plt.savefig("output/" + "{}_n_ts{}_ptype{}".format(example_name, n_ts, p_type) + ".png")