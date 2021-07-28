import numpy as np
import matplotlib.pyplot as plt

x = [
160,
320,
640,
1080,
2160,
5120,
10240
















]
sum_norm = [
4.143671700023951e-10,
5.726212731813453e-10,
3.42488751700192e-10,
1.971339248184725e-10,
1.6610215482488401e-10,
1.1835179225373598e-10,
9.245265933305356e-11
# 1475.911,
# 2951.835012234831,
# 5903.665777347067,
# 11807.334553957924,
# 23614.669107915797,
# 47229.33821583199,
# 94458.67643166312
]
obj_r = [
0.195,
0.14,
0.0186,
0.00646,
0.001304,
0.000401,
9.61976E-05

]

inf_norm = [6.07E+00,
6.02E+00,
6.01E+00,
6.01E+00,
6.01E+00,
6.01E+00,
6.01E+00,
6.01E+00,
6.01E+00,
6.01E+00,
6.01E+00

]

plt.figure()
plt.plot(np.array(x), np.array(sum_norm), '-o', label='squared_L2_norm')
plt.xlabel("Rounding Time Steps")
plt.ylabel("Squared Penalized Term")
plt.legend()
plt.savefig("SPIN4_rounding_penalized_p1.png")
#
exit()
plt.figure()
plt.plot([np.log10(x_) for x_ in x], np.log10(np.array(obj_r)), '-o', label='Obj-R')
plt.xlabel("Log of Rounding Time Steps")
plt.ylabel("Log of Objective Value after Rounding")
plt.legend()
plt.savefig("SPIN4_obj_r_sur_p1_new.png")
