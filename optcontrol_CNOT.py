import datetime
import numpy as np
import matplotlib.pyplot as plt

from qutip import Qobj, identity, sigmax, sigmaz, sigmay, tensor
from qutip.qip.operations.gates import cnot
import qutip.logging_utils as logging
import qutip.control.pulseoptim as cpo
import qutip.control.pulsegen as pulsegen

logger = logging.get_logger()
# Set this to None or logging.WARN for 'quiet' execution
log_level = logging.INFO
# QuTiP control modules
# a two-qubit system with target control mode as CNOT gate
example_name = 'CNOT'

# The control Hamiltonians
H_c = [tensor(sigmax(), identity(2)), tensor(sigmay(), identity(2))]
# Drift Hamiltonian
H_d = tensor(sigmax(), sigmax()) + tensor(sigmay(), sigmay()) + tensor(sigmaz(), sigmaz())
# start point for the gate evolution
U_0 = identity(4)
# Target for the gate evolution
U_targ = cnot()

# Time allowed for the evolution
evo_time = 1
# Number of time slots
n_ts = 20 * evo_time
# Number of controllers
n_ctrls = len(H_c)

# Fidelity error target
fid_err_targ = 1e-10
# Maximum iterations for the optisation algorithm
max_iter = 500
# Maximum (elapsed) time allowed in seconds
max_wall_time = 600
# Minimum gradient (sum of gradients squared)
# as this tends to 0 -> local minima has been found
min_grad = 1e-10

# Set initial state
# pulse type alternatives: RND|ZERO|LIN|SINE|SQUARE|SAW|TRIANGLE|
p_type = 'RND'
offset = 0
# lower bound and upper bound of initial value
init_lb = 0
init_ub = 1
obj_type = "TRACEDIFF"
# Set output files
f_ext = "{}_n_ts{}_ptype{}.txt".format(example_name, n_ts, p_type)

# Build the optimizer
optim = cpo.create_pulse_optimizer(H_d, H_c, U_0, U_targ, n_ts, evo_time,
                                   amp_lbound=0, amp_ubound=1,
                                   fid_err_targ=fid_err_targ, min_grad=min_grad,
                                   max_iter=max_iter, max_wall_time=max_wall_time, dyn_type='UNIT',
                                   fid_type=obj_type, phase_option="PSU",
                                   init_pulse_type=p_type, init_pulse_params={"offset": offset},
                                   log_level=log_level, gen_stats=True)

# Initialization
dyn = optim.dynamics
p_gen = optim.pulse_generator
p_gen = pulsegen.create_pulse_gen(p_type, dyn)

# Generate different initial pulses for each of the controls
init_amps = np.zeros([n_ts, n_ctrls])
for j in range(n_ctrls):
    p_gen.lbound = init_lb
    p_gen.ubound = init_ub
    p_gen.offset = offset
    init_amps[:, j] = p_gen.gen_pulse()
dyn.initialize_controls(init_amps)

# Run the optimization
result = optim.run_optimization()

# Report the results
result.stats.report()
report = open("output/" + "{}_evotime{}_n_ts{}_ptype{}_offset{}_obj{}".format(
    example_name, evo_time, n_ts, p_type, offset, obj_type) + ".log", "w+")
print("Final evolution\n{}\n".format(result.evo_full_final), file=report)
print("********* Summary *****************", file=report)
print("Final fidelity error {}".format(result.fid_err), file=report)
print("Final gradient normal {}".format(result.grad_norm_final), file=report)
print("Terminated due to {}".format(result.termination_reason), file=report)
print("Number of iterations {}".format(result.num_iter), file=report)
print("Completed in {} HH:MM:SS.US".format(
    datetime.timedelta(seconds=result.wall_time)), file=report)

# Plot the results
fig1 = plt.figure()
ax1 = fig1.add_subplot(2, 1, 1)
ax1.set_title("Initial control amps")
# ax1.set_xlabel("Time")
ax1.set_ylabel("Control amplitude")
for j in range(n_ctrls):
    ax1.step(result.time,
             np.hstack((result.initial_amps[:, j], result.initial_amps[-1, j])),
             where='post')

ax2 = fig1.add_subplot(2, 1, 2)
ax2.set_title("Optimised Control Sequences")
ax2.set_xlabel("Time")
ax2.set_ylabel("Control amplitude")
for j in range(n_ctrls):
    ax2.step(result.time,
             np.hstack((result.final_amps[:, j], result.final_amps[-1, j])),
             where='post')
plt.tight_layout()
plt.savefig("output/" + "{}_evotime{}_n_ts{}_ptype{}_offset{}_obj{}".format(
    example_name, evo_time, n_ts, p_type, offset, obj_type) + ".png")

