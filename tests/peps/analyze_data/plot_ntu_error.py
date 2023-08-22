import matplotlib.pyplot as plt
import numpy as np

nfont = 14


plt.rcParams["font.size"] = nfont
plt.rcParams['font.family'] = "Arial"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{braket}')

# Importing the data from the first file
ne_D4 = 'NTU_error_sample_1_shape_rectangle_Nx_4_Ny_4_boundary_finite_purification_True_fixed_bd_0.0_target_beta_4.000_optimal_two-step_Ds_4_U_8.00_MU_UP_0.00000_MU_DN_0.00000_T_UP_1.00_T_DN_1.00_Z2.txt'
ne_D6 = 'NTU_error_sample_1_shape_rectangle_Nx_4_Ny_4_boundary_finite_purification_True_fixed_bd_0.0_target_beta_4.000_optimal_one-step_Ds_6_U_8.00_MU_UP_0.00000_MU_DN_0.00000_T_UP_1.00_T_DN_1.00_Z2.txt'

data_4 = np.loadtxt(ne_D4)
data_6 = np.loadtxt(ne_D6)

x1, x2 = 2401, 3200
x3, x4 = 4801, 5400
x5, x6 = 6401, 7200
x7, x8 = 0, 800

# Plotting the data
plt.plot(data_4[x1:x2:20,0], (data_4[x1:x2:20,1]+data_4[x1:x2:20,2])/0.01, label=r'instance 1 $D=4$')
plt.plot(data_6[x1:x2:20,0], (data_6[x1:x2:20,1]+data_6[x1:x2:20,2])/0.01, label=r'instance 1 $D=6$')

plt.plot(data_4[x3:x4:20,0], (data_4[x3:x4:20,1]+data_4[x3:x4:20,2])/0.01, label=r'instance 2 $D=4$')
plt.plot(data_6[x7:x8:20,0], (data_6[x7:x8:20,1]+data_6[x7:x8:20,2])/0.01, label=r'instance 2 $D=6$')

plt.plot(data_4[x5:x6:20,0], (data_4[x5:x6:20,1]+data_4[x5:x6:20,2])/0.01, label=r'instance 3 $D=4$')
plt.plot(data_6[1601:2400:20,0], (data_6[1601:2400:20,1]+data_6[1601:2400:20,2])/0.01, label=r'instance 3 $D=6$')

# Adding labels and title
plt.xlabel(r'$\beta$')
plt.ylabel(r'ntu err $\delta/d\beta$')


# Adding a legend
plt.legend()

# Displaying the plot
plt.show()
