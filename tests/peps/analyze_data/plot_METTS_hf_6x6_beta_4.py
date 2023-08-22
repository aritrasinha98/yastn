import numpy as np
import matplotlib.pyplot as plt

def running_average(x):
    cumsum = np.cumsum(x)
    return cumsum / (np.arange(len(x)) + 1)

def running_std_deviation(combined_data, x):
    running_avg = running_average(combined_data[:x])

    running_sum_squared_diff = np.zeros_like(running_avg)
    running_variance = np.zeros_like(running_avg)
    running_std_dev = np.zeros_like(running_avg)

    for i in range(x):
        squared_diff = (combined_data[i] - running_avg[i]) ** 2
        if i == 0:
            running_sum_squared_diff[i] = squared_diff
            running_variance[i] = 0.0
        else:
            running_sum_squared_diff[i] = running_sum_squared_diff[i - 1] + squared_diff
            running_variance[i] = running_sum_squared_diff[i] / i

        running_std_dev[i] = np.sqrt(running_variance[i])

    return running_std_dev


# Create a figure with two subplots

nfont = 20


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
file_name1 = f"sz_basis_sample_1_observables_from_METTS_Hubbard_hf_target_beta_4.0_shape_rectangle_Nx_6_Ny_6_boundary_finite_purification_True_fixed_bd_0.0_target_beta_4.000_optimal_one-step_Ds_4_U_8.00_MU_UP_0.00000_MU_DN_0.00000_T_UP_1.00_T_DN_1.00_Z2.txt"
file_name2 = f"sz_basis_sample_2_observables_from_METTS_Hubbard_hf_target_beta_4.0_shape_rectangle_Nx_6_Ny_6_boundary_finite_purification_True_fixed_bd_0.0_target_beta_4.000_optimal_one-step_Ds_4_U_8.00_MU_UP_0.00000_MU_DN_0.00000_T_UP_1.00_T_DN_1.00_Z2.txt"
file_name3 = f"sz_basis_sample_3_observables_from_METTS_Hubbard_hf_target_beta_4.0_shape_rectangle_Nx_6_Ny_6_boundary_finite_purification_True_fixed_bd_0.0_target_beta_4.000_optimal_one-step_Ds_4_U_8.00_MU_UP_0.00000_MU_DN_0.00000_T_UP_1.00_T_DN_1.00_Z2.txt"
file_name4 = f"sz_basis_sample_4_observables_from_METTS_Hubbard_hf_target_beta_4.0_shape_rectangle_Nx_6_Ny_6_boundary_finite_purification_True_fixed_bd_0.0_target_beta_4.000_optimal_one-step_Ds_4_U_8.00_MU_UP_0.00000_MU_DN_0.00000_T_UP_1.00_T_DN_1.00_Z2.txt"

# Load the data from the file
data1 = np.loadtxt(file_name1)
data2 = np.loadtxt(file_name2)
data3 = np.loadtxt(file_name3)
data4 = np.loadtxt(file_name4)


nsm =108
tot_nsm = 3*nsm
combined_energy_data = np.empty(tot_nsm, dtype=data1.dtype)
combined_energy_data[0:tot_nsm:3] = data1[0:nsm,1]
combined_energy_data[1:tot_nsm:3] = data3[0:nsm,1]
combined_energy_data[2:tot_nsm:3] = data4[0:nsm,1]


combined_do_data = np.empty(tot_nsm, dtype=data1.dtype)
combined_do_data[0:tot_nsm:3] = data1[0:nsm,2]
combined_do_data[1:tot_nsm:3] = data3[0:nsm,2]
combined_do_data[2:tot_nsm:3] = data4[0:nsm,2]


running_energy = running_average(combined_energy_data)
running_do = running_average(combined_do_data)
running_std_dev_energy = running_std_deviation(combined_energy_data, tot_nsm)  # Replace x with the number of samples you want
running_std_dev_do = running_std_deviation(combined_do_data, tot_nsm)  # Replace x with the number of samples you want


# Plot the energy data

ax[0].plot(running_energy,label=r'$D=4$')
ax[1].plot(running_do,label=r'$D=4$')
ax[0].fill_between(np.arange(len(running_energy)), running_energy - running_std_dev_energy, running_energy + running_std_dev_energy, alpha=0.4)
ax[1].fill_between(np.arange(len(running_do)), running_do - running_std_dev_do, running_do + running_std_dev_do, alpha=0.4)

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



# Convert lists to arrays and compute running average
nsm = 67
"""avg_energy = (1/3)*running_average(data2[0:nsm,1]+data3[0:nsm,1]+data4[0:nsm,1])
avg_do = (1/3)*running_average(data2[0:nsm,2]+data3[0:nsm,2]+data4[0:nsm,2])"""
tot_nsm = 3*nsm
combined_energy_data = np.empty(tot_nsm, dtype=data1.dtype)
combined_energy_data[0:tot_nsm:3] = data2[0:nsm,1]
combined_energy_data[1:tot_nsm:3] = data3[0:nsm,1]
combined_energy_data[2:tot_nsm:3] = data4[0:nsm,1]

combined_do_data = np.empty(tot_nsm, dtype=data1.dtype)
combined_do_data[0:tot_nsm:3] = data2[0:nsm,2]
combined_do_data[1:tot_nsm:3] = data3[0:nsm,2]
combined_do_data[2:tot_nsm:3] = data4[0:nsm,2]

running_energy = running_average(combined_energy_data)
running_do = running_average(combined_do_data)
running_std_dev_energy = running_std_deviation(combined_energy_data, tot_nsm)  # Replace x with the number of samples you want
running_std_dev_do = running_std_deviation(combined_do_data, tot_nsm)  # Replace x with the number of samples you want

ax[0].plot(running_energy, label=r'$D=5$')
ax[1].plot(running_do,label=r'$D=5$')
ax[0].fill_between(np.arange(len(running_energy)), running_energy - running_std_dev_energy, running_energy + running_std_dev_energy, alpha=0.4)
ax[1].fill_between(np.arange(len(running_do)), running_do - running_std_dev_do, running_do + running_std_dev_do, alpha=0.4)

# Initialize empty lists to hold all the data

# Loop over the 5 different files
    # Define the file name
file_name1 = f"sz_basis_sample_1_observables_from_METTS_Hubbard_hf_target_beta_4.0_shape_rectangle_Nx_6_Ny_6_boundary_finite_purification_True_fixed_bd_0.0_target_beta_4.000_optimal_one-step_Ds_6_U_8.00_MU_UP_0.00000_MU_DN_0.00000_T_UP_1.00_T_DN_1.00_Z2.txt"
file_name2 = f"sz_basis_sample_2_observables_from_METTS_Hubbard_hf_target_beta_4.0_shape_rectangle_Nx_6_Ny_6_boundary_finite_purification_True_fixed_bd_0.0_target_beta_4.000_optimal_one-step_Ds_6_U_8.00_MU_UP_0.00000_MU_DN_0.00000_T_UP_1.00_T_DN_1.00_Z2.txt"
file_name3 = f"sz_basis_sample_3_observables_from_METTS_Hubbard_hf_target_beta_4.0_shape_rectangle_Nx_6_Ny_6_boundary_finite_purification_True_fixed_bd_0.0_target_beta_4.000_optimal_one-step_Ds_6_U_8.00_MU_UP_0.00000_MU_DN_0.00000_T_UP_1.00_T_DN_1.00_Z2.txt"
file_name4 = f"sz_basis_sample_4_observables_from_METTS_Hubbard_hf_target_beta_4.0_shape_rectangle_Nx_6_Ny_6_boundary_finite_purification_True_fixed_bd_0.0_target_beta_4.000_optimal_one-step_Ds_6_U_8.00_MU_UP_0.00000_MU_DN_0.00000_T_UP_1.00_T_DN_1.00_Z2.txt"
file_name5 = f"sz_basis_sample_5_observables_from_METTS_Hubbard_hf_target_beta_4.0_shape_rectangle_Nx_6_Ny_6_boundary_finite_purification_True_fixed_bd_0.0_target_beta_4.000_optimal_one-step_Ds_6_U_8.00_MU_UP_0.00000_MU_DN_0.00000_T_UP_1.00_T_DN_1.00_Z2.txt"

# Load the data from the file
data1 = np.loadtxt(file_name1)
data2 = np.loadtxt(file_name2)
data3 = np.loadtxt(file_name3)
data4 = np.loadtxt(file_name4)
data5 = np.loadtxt(file_name5)


# Convert lists to arrays and compute running average
nsm = 99

tot_nsm = 5*nsm
combined_energy_data = np.empty(tot_nsm, dtype=data1.dtype)
combined_energy_data[0:tot_nsm:5] = data1[0:nsm,1]
combined_energy_data[1:tot_nsm:5] = data2[0:nsm,1]
combined_energy_data[2:tot_nsm:5] = data3[0:nsm,1]
combined_energy_data[3:tot_nsm:5] = data4[0:nsm,1]
combined_energy_data[4:tot_nsm:5] = data5[0:nsm,1]


combined_do_data = np.empty(tot_nsm, dtype=data1.dtype)
combined_do_data[0:tot_nsm:5] = data1[0:nsm,2]
combined_do_data[1:tot_nsm:5] = data2[0:nsm,2]
combined_do_data[2:tot_nsm:5] = data3[0:nsm,2]
combined_do_data[3:tot_nsm:5] = data4[0:nsm,2]
combined_do_data[4:tot_nsm:5] = data5[0:nsm,2]


running_energy = running_average(combined_energy_data)
running_do = running_average(combined_do_data)

running_std_dev_energy = running_std_deviation(combined_energy_data, tot_nsm)  # Replace x with the number of samples you want
running_std_dev_do = running_std_deviation(combined_do_data, tot_nsm)  # Replace x with the number of samples you want
ax[0].plot(running_energy,label=r'$D=6$')
ax[1].plot(running_do,label=r'$D=6$')
ax[0].fill_between(np.arange(len(running_energy)), running_energy - running_std_dev_energy, running_energy + running_std_dev_energy, alpha=0.4)
ax[1].fill_between(np.arange(len(running_do)), running_do - running_std_dev_do, running_do + running_std_dev_do, alpha=0.4)

ax[0].set_ylabel(r'$E$')

ax[1].set_ylabel(r'$\langle n_{\uparrow}n_{\downarrow} \rangle$')


# Add the horizontal lines representing the iPEPS data
ax[0].axhline(y=-0.39716257295481894, color='g', linestyle='--', label=r'purification $D=8$')
ax[1].axhline(y=0.04362828192614411, color='g', linestyle='--', label=r'purification $D=8$')
ax[0].axhline(y=-0.3960287537163052, color='cyan', linestyle='--', label=r'purification $D=12$')
ax[1].axhline(y=0.043722543825832086, color='cyan', linestyle='--', label=r'purification $D=12$')
ax[0].axhline(y=-0.398734667938983, color='b', linestyle='--', label=r'purification $D=16$')
ax[1].axhline(y=0.04370055987561957, color='b', linestyle='--', label=r'purification $D=16$')
ax[0].axhline(y=-0.4003204680275452, color='r', linestyle='--', label=r'purification $D=20$')
ax[1].axhline(y=0.04367352065519445, color='r', linestyle='--', label=r'purification $D=20$')
ax[0].set_xlim([0,560])
ax[1].set_xlim([0,560])
# Add legends
ax[0].legend(bbox_to_anchor=(0., 1.1, 1., .11), loc='lower left', ncol=3, mode="expand", borderaxespad=0.)
plt.tight_layout()
plt.show()
