import numpy as np
import matplotlib.pyplot as plt

def running_average(x):
    cumsum = np.cumsum(x)
    return cumsum / (np.arange(len(x)) + 1)

# Create a figure with two subplots

nfont = 12


plt.rcParams["font.size"] = nfont
plt.rcParams['font.family'] = "Arial"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{braket}')
fig, ax = plt.subplots(2, 1, figsize=(10, 8))

# Initialize empty lists to hold all the data

# Loop over the 5 different files
    # Define the file name
file_name1 = f"sz_basis_sample_1_observables_from_METTS_Hubbard_hf_target_beta_4.0_shape_rectangle_Nx_6_Ny_6_boundary_finite_purification_True_fixed_bd_0.0_target_beta_4.000_optimal_one-step_Ds_5_U_8.00_MU_UP_0.00000_MU_DN_0.00000_T_UP_1.00_T_DN_1.00_Z2.txt"
file_name2 = f"sz_basis_sample_2_observables_from_METTS_Hubbard_hf_target_beta_4.0_shape_rectangle_Nx_6_Ny_6_boundary_finite_purification_True_fixed_bd_0.0_target_beta_4.000_optimal_one-step_Ds_5_U_8.00_MU_UP_0.00000_MU_DN_0.00000_T_UP_1.00_T_DN_1.00_Z2.txt"
file_name3 = f"sz_basis_sample_3_observables_from_METTS_Hubbard_hf_target_beta_4.0_shape_rectangle_Nx_6_Ny_6_boundary_finite_purification_True_fixed_bd_0.0_target_beta_4.000_optimal_one-step_Ds_5_U_8.00_MU_UP_0.00000_MU_DN_0.00000_T_UP_1.00_T_DN_1.00_Z2.txt"
file_name4 = f"sz_basis_sample_4_observables_from_METTS_Hubbard_hf_target_beta_4.0_shape_rectangle_Nx_6_Ny_6_boundary_finite_purification_True_fixed_bd_0.0_target_beta_4.000_optimal_one-step_Ds_5_U_8.00_MU_UP_0.00000_MU_DN_0.00000_T_UP_1.00_T_DN_1.00_Z2.txt"
file_name5 = f"sz_basis_sample_5_observables_from_METTS_Hubbard_hf_target_beta_4.0_shape_rectangle_Nx_6_Ny_6_boundary_finite_purification_True_fixed_bd_0.0_target_beta_4.000_optimal_one-step_Ds_5_U_8.00_MU_UP_0.00000_MU_DN_0.00000_T_UP_1.00_T_DN_1.00_Z2.txt"

# Load the data from the file
data1 = np.loadtxt(file_name1)
data2 = np.loadtxt(file_name2)
data3 = np.loadtxt(file_name3)
data4 = np.loadtxt(file_name4)
data5 = np.loadtxt(file_name5)

sample_1_energy = running_average(data1[:,1])
sample_1_do = running_average(data1[:,2])
sample_2_energy = running_average(data2[:,1])
sample_2_do = running_average(data2[:,2])
sample_3_energy = running_average(data3[:,1])
sample_3_do = running_average(data3[:,2])
sample_4_energy = running_average(data4[:,1])
sample_4_do = running_average(data4[:,2])
sample_5_energy = running_average(data5[:,1])
sample_5_do = running_average(data5[:,2])

# Convert lists to arrays and compute running average
nsm = 17
avg_energy = (1/5)*running_average(data1[0:nsm,1]+data2[0:nsm,1]+data3[0:nsm,1]+data4[0:nsm,1]+data5[0:nsm,1])
avg_do = (1/5)*running_average(data1[0:nsm,2]+data2[0:nsm,2]+data3[0:nsm,2]+data4[0:nsm,2]+data5[0:nsm,2])

# Plot the energy data
ax[0].plot(sample_1_energy[0:nsm], label=r'sample 1, D=5')
ax[0].plot(sample_2_energy[0:nsm], label=r'sample 2, D=5')
ax[0].plot(sample_3_energy[0:nsm], label=r'sample 3, D=5')
ax[0].plot(sample_4_energy[0:nsm], label=r'sample 4, D=5')
ax[0].plot(sample_5_energy[0:nsm], label=r'sample 5, D=5')
ax[0].plot(avg_energy, 'o-',label=r'average D=5')
ax[0].set_ylabel(r'$E$')
ax[0].set_title(r'Half-filling $6\times 6 \quad \beta = 4$')


# Plot the double occupancy data
ax[1].plot(sample_1_do[0:nsm], label=r'sample 1, D=5')
ax[1].plot(sample_2_do[0:nsm], label=r'sample 2, D=5')
ax[1].plot(sample_3_do[0:nsm], label=r'sample 3, D=5')
ax[1].plot(sample_4_do[0:nsm], label=r'sample 4, D=5')
ax[1].plot(sample_5_do[0:nsm], label=r'sample 5, D=5')
ax[1].plot(avg_do, 'o-',label=r'avg D=5')
ax[1].set_ylabel(r'$\langle n_{\uparrow}n_{\downarrow} \rangle$')


# Add the horizontal lines representing the iPEPS data
ax[0].axhline(y=-0.4003204680275452, color='r', linestyle='--', label=r'purification $D=20$')
ax[1].axhline(y=0.04367352065519445, color='r', linestyle='--', label=r'purification $D=20$')

# Add legends
ax[0].legend(bbox_to_anchor=(0., 1.1, 1., .11), loc='lower left', ncol=4, mode="expand", borderaxespad=0.)
plt.tight_layout()
plt.show()
