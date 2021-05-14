# Quantum-Pulse-OPT
Quantum-Pulse-OPT is the code for the quantum pulse optimization problem. 

## Install dependencies

You need to have Python with version 3.6 or later. You need also to install the package QuTip by the following command. 

```
pip install qutip
```

## Examples

There are two examples, the Ising model with one single controller and the CNOT model with two controllers. 

### Ising model

This model includes only one Hamiltonian controller. The users can run this example by the following command line. 

```
python example/Ising.py
```

To do the rounding, please set 

```python
if_rounding = True
```

The users can set rounding parameters by the following code. 

```python
rounding_type = "BnB" # Another choice: "SUR"
min_up_time = 10
```

To use the warm start, please set

```python
if_warm_start = True
```

The users can choose the warm start control you want from a .csv file by setting the variable initial_control as the file name. Here is an example. 

```python
initial_control = "../control/" + "{}_evotime{}_n_ts{}_ptype{}_offset{}_obj{}".format(example_name, 1, 16, "ZERO", 0.5, obj_type) + ".csv"
```

The users can change the evolution time and number of time steps by setting the following variables. 

```python
# change the value 
evo_time = 1
n_ts = 16
```

### CNOT model

This model include two Hamiltonian controllers to estimate the CNOT gate. The users can run this example by the following command line. 

```
python example/CNOT.py
```

To do the rounding, please set 

```python
if_rounding = True
```

The users can set rounding parameters by the following code. 

```python
rounding_type = "BnB" # Another choice: "SUR"
min_up_time = 10
```

To use the warm start, please set

```python
if_warm_start = True
```

The users can choose the warm start control you want from a .csv file by setting the variable initial_control as the file name. Here is an example. 

```python
initial_control = "../control/" + "{}_evotime{}_n_ts{}_ptype{}_offset{}_obj{}".format(example_name, 10, 200, "ZERO", 0.5, obj_type) + ".csv"
```

The users can add a constraint indicating that the summation of all the control variables should be 1 at each time step by setting 

```python
sum_cons_1 = True
```

The users can change the evolution time and number of time steps by setting the following variables. 

```python
# change the value 
evo_time = 20
n_ts = 400
```

## Build New Example

Here are the following steps of solving the quantum optimal control by our code based on the QuTip. 

Before starting, please import the needed functions. 

```python
import os
from qutip import Qobj, identity, sigmax, sigmaz, sigmay, tensor
from qutip.qip.operations.gates import cnot

from optcontrol import optcontrol
from rounding import rounding
from evolution import time_evolution, compute_obj
```

##### Step 1: set Hamiltonian matrices

```python
# The control Hamiltonians
H_c = [tensor(sigmax(), identity(2)), tensor(sigmay(), identity(2))]
# Drift Hamiltonian
H_d = tensor(sigmax(), sigmax()) + tensor(sigmay(), sigmay()) + tensor(sigmaz(), sigmaz())
# start point for the gate evolution
X_0 = identity(4)
# Target for the gate evolution
X_targ = cnot()
```

##### Step 2: set number of time slots and time interval

```python
# Number of time slots
n_ts = 40
# Time allowed for the evolution
evo_time = 2
```

##### Step 3: set convergence parameters

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

##### Step 4: set initial values of control variables

```python
p_type = "ZERO"
offset = 0.5
obj_type = "UNIT"
initial_control = None  # no warm start

# if you want to use warm start
if_warm_start = 1
if if_warm_start:
    p_type = "WARM"
    offset = 0
    initial_control = "../control/" + "{}_evotime{}_n_ts{}_ptype{}_offset{}_obj{}".format(
        example_name, 10, 200, "ZERO", 0.5, obj_type) + ".csv"
```

##### Step 5: set the output files

```python
# output file of the numerical results
output_num = "../output/" + "{}_evotime{}_n_ts{}_ptype{}_offset{}_obj{}".format(
    example_name, evo_time, n_ts, p_type, offset, obj_type) + ".log"
# output file of the figures
output_fig = "../output/" + "{}_evotime{}_n_ts{}_ptype{}_offset{}_obj{}".format(
    example_name, evo_time, n_ts, p_type, offset, obj_type) + ".png"
# output file of the controls
output_control = "../control/" + "{}_evotime{}_n_ts{}_ptype{}_offset{}_obj{}".format(
    example_name, evo_time, n_ts, p_type, offset, obj_type) + ".csv"
```

##### Step 6: run optimization and output results

```python
# choose to add the summation as 1 constraint or not
sum_cons_1 = True
# run the optimization
optcontrol(example_name, H_d, H_c, X_0, X_targ, n_ts, evo_time, p_type, initial_control, output_num, output_fig, output_control, sum_cons_1, fid_err_targ, max_iter, max_wall_time, min_grad, offset)
```

##### Step 7: do the rounding and compute the error (optional)

```python
if_rounding = True
if if_rounding:
    # set the rounding type and minimized up time
    rounding_type = "BnB"
    min_up_time = 1

    # do the rounding
    bin_amps = rounding(output_control, rounding_type, min_up_time)
	
    # compute and output the objective value after rounding
    rounding_result = time_evolution(
        H_d.full(), [h_c.full() for h_c in H_c], n_ts, evo_time, bin_amps.T, X_0.full(), sum_cons_1)
    f = open(output_control.split(".csv")[0] + "_binary_" + str(min_up_time) + ".log", "w+")
    print("Rounding result: ", file=f)
    print(rounding_result, file=f)
    print("Final objective value: ", file=f)
    print(compute_obj(X_targ, rounding_result), file=f)
```

