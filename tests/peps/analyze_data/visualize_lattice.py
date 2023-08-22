import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio
import os

plt.rcParams['text.usetex'] = True

def plot_state(ax, x, y, state):
    circle = patches.Circle((x, y), radius=0.12, edgecolor='black', facecolor='gray' if state == 3 else 'white', fill=True, linewidth=2)
    ax.add_patch(circle)

    if state == 0:  # Spin up
        for i in range(3):
            arrow = patches.FancyArrow(x, y-0.1-i*0.01, 0, 0.001, color=f'red', alpha=0.7-i*0.2, width=0.015, head_width=0.1, head_length=0.2, linewidth=2)
            ax.add_patch(arrow)

    elif state == 1:  # Spin down
        for i in range(3):
            arrow = patches.FancyArrow(x, y+0.1+i*0.01, 0, -0.001, color='blue', alpha=0.7-i*0.2, width=0.015, head_width=0.1, head_length=0.2, linewidth=2)
            ax.add_patch(arrow)

    elif state == 2:  # Double occupancy
        for i in range(3):
            arrow_up = patches.FancyArrow(x-0.05, y-0.1-i*0.01, 0, 0.001, color='red', alpha=0.7-i*0.2, width=0.015, head_width=0.1, head_length=0.2, linewidth=2)
            arrow_down = patches.FancyArrow(x+0.05, y+0.1+i*0.01, 0, -0.01, color='blue', alpha=0.7-i*0.2, width=0.015, head_width=0.1, head_length=0.2, linewidth=2)
            ax.add_patch(arrow_up)
            ax.add_patch(arrow_down)


def initialize_plot():
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 4)
    ax.set_aspect('equal')
    plt.xticks(range(4))
    plt.yticks(range(4))
    ax.axis('off')  # Turn off axis lines
    return fig, ax

def main():
    # Load the data
    loaded_file = np.load("samples_generated_from_sample_3_shape_rectangle_Nx_4_Ny_4_boundary_finite_purification_True_fixed_bd_0.0_target_beta_4.000_optimal_one-step_Ds_6_U_8.00_MU_UP_0.00000_MU_DN_0.00000_T_UP_1.00_T_DN_1.00_Z2.npy", allow_pickle=True).item()

    # For each iteration
    for iteration, data in loaded_file.items():
        fig, ax = initialize_plot()
        for key, value in data.items():
            y, x = key[0], key[1]  # Swap the x and y coordinates
            plot_state(ax, x, y, value)
        plt.title(r'$4 \times 4 \ D=6 \ \beta=4 \ \mathrm{Iteration:}$' + f' {iteration}', fontsize=30)
        plt.savefig(f"Iteration_{iteration}.png", dpi=50)  # Save the figure with iteration number
        plt.close(fig)  # Close the figure to free memory
        
    image_directory = './'
    image_files = [image_directory + f"Iteration_{i}.png" for i in range(1, 41)]

    # Read the images using imageio
    images = [imageio.imread(file) for file in image_files]

    # Create a gif from the images
    output_file = 'movie.gif'  # Name of the gif file
    imageio.mimsave(output_file, images, fps=1)

if __name__ == "__main__":
    main()
