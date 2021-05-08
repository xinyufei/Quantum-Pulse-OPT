# Quantum-Pulse-OPT
Quantum-Pulse-OPT is the code for the quantum pulse optimization problem. 

#### Install dependencies

You need to have Python with version 3.6 or later. You need also to install the package QuTip by the following command. 

```
pip install qutip
```

#### Guides

Here are the following steps of solving quantum optimal control by QuTip. 

##### Step 1: generate matrix

You can generate different matrices according to different samples. 

```python
generate_Jij(n)
C_mat = get_ham(n, True)
np.savetxt('output/C_mat' + str(n) + '.csv', C_mat)
B_mat = get_ham(n, False)
np.savetxt('output/B_mate' + str(n) + '.csv', B_mat)
```

##### Step 2: set corresponding matrices

```python
# Control Hamiltonian
H_c = [Qobj(B_mat - C_mat)]
# Drift Hamiltonian
H_d = Qobj(C_mat)
# start point
Ground_B = Qobj(B_mat).groundstate()[1]
X_0 = Ground_B
# Target
Ground_C = Qobj(C_mat).groundstate()[1]
X_targ = Ground_C
```

##### Step 3: set number of time slots and time interval

```python
# Number of time slots
n_ts = 40
# Time allowed for the evolution
evo_time = 2
```

##### Step 4: set convergence parameters

```python
# Fidelity error target
fid_err_targ = 1e-10
# Maximum iterations for the optisation algorithm
max_iter = 500
# Maximum (elapsed) time allowed in seconds
max_wall_time = 120
# Minimum gradient (sum of gradients squared)
# as this tends to 0 -> local minima has been found
min_grad = 1e-50
```

##### Step 5: set initial values of u

```python
# set it as 'ZERO' for constant initail value
p_type = 'RND' 
# offset of initial value
offset = 0.5
# lower bound and upper bound of initial value
init_lb = 0
init_ub = 1
```

##### Step 6: build optimizer

```python
# Build the optimizer 
# amp_lbound, amp_ubound: lower bound & upper bound
optim = cpo.create_pulse_optimizer(H_d, H_c, X_0, X_targ, n_ts, evo_time, amp_lbound=0, amp_ubound=1, fid_err_targ=fid_err_targ, min_grad=min_grad, max_iter=max_iter, max_wall_time=max_wall_time, dyn_type='GEN_MAT', fid_type='TRACEDIFF', init_pulse_type=p_type, init_pulse_params={"offset": offset}, log_level=log_level, gen_stats=True)
```

##### Step 7: run optimization and output results

```python
# Run the optimization
result = optim.run_optimization()
```

