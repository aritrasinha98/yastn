# Here's a revised version of your script with the suggested improvements:

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio
import os

plt.rcParams['text.usetex'] = True

def plot_state(ax, x, y, state):
    """
    Plot the state of the lattice at position (x, y) with the given state.
    """
    circle = patches.Circle((x, y), radius=0.2, edgecolor='black', facecolor='gray' if state == 3 else 'white', fill=True, linewidth=2)
    ax.add_patch(circle)

    if state == 0:  # Spin up
        for i in range(3):
            arrow = patches.FancyArrow(x, y-0.1-i*0.01, 0, 0.001, color=f'red', alpha=0.7-i*0.2, width=0.015, head_width=0.14, head_length=0.2, linewidth=2)
            ax.add_patch(arrow)

    elif state == 1:  # Spin down
        for i in range(3):
            arrow = patches.FancyArrow(x, y+0.1+i*0.01, 0, -0.001, color='blue', alpha=0.7-i*0.2, width=0.015, head_width=0.14, head_length=0.2, linewidth=2)
            ax.add_patch(arrow)

    elif state == 2:  # Double occupancy
        for i in range(3):
            arrow_up = patches.FancyArrow(x-0.05, y-0.1-i*0.01, 0, 0.001, color='red', alpha=0.7-i*0.2, width=0.015, head_width=0.14, head_length=0.2, linewidth=2)
            arrow_down = patches.FancyArrow(x+0.05, y+0.1+i*0.01, 0, -0.01, color='blue', alpha=0.7-i*0.2, width=0.015, head_width=0.14, head_length=0.2, linewidth=2)
            ax.add_patch(arrow_up)
            ax.add_patch(arrow_down)

def initialize_plot():
    """
    Initialize the plot with the desired settings.
    """
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 4)
    ax.set_aspect('equal')
    plt.xticks(range(4))
    plt.yticks(range(4))
    ax.axis('off')  # Turn off axis lines
    return fig, ax

def generate_plots(data_file):
    """
    Generate plots for each iteration in the data file.
    """
    # Check if the data file exists
    if not os.path.isfile(data_file):
        print(f"Data file not found: {data_file}")
        return []
    
    # Load the data
    loaded_file = np.load(data_file, allow_pickle=True).item()

    image_files = []

    # For each iteration
    for iteration, data in loaded_file.items():
        fig, ax = initialize_plot()
        for key, value in data.items():
            y, x = key[0], key[1]  # Swap the x and y coordinates
            plot_state(ax, x, y, value)
        plt.title(r'$2D \ FHM  \ U=8 \ HF \ 4 \times 4 \ D=6 \ \beta = 6 \ \mathrm{Iteration:}$' + f' {iteration}', fontsize=30)
        image_file = f"Iteration_{iteration}.png"
        plt.savefig(image_file, dpi=100)  # Save the figure with iteration number
        plt.close(fig)  # Close the figure to free memory
        
        image_files.append(image_file)
    
    return image_files

def create_gif(image_files, output_file, fps=1):
    """
    Create a gif from the list of image files.
    """
    # Check if there are any image files
    if not image_files:
        print("No image files to create gif.")
        return
    
    # Read the images using imageio
    images = [imageio.imread(file) for file in image_files]

    # Create the gif
    imageio.mimsave(output_file, images, fps=fps)

def main():
    data_file = "samples_generated_from_sample_2_shape_rectangle_Nx_4_Ny_4_boundary_finite_purification_True_fixed_bd_0.0_target_beta_6.000_optimal_one-step_Ds_6_U_8.00_MU_UP_0.00000_MU_DN_0.00000_T_UP_1.00_T_DN_1.00_Z2.npy"
    output_file = 'movie.gif'  # Name of the gif file

    # Generate the plots
    image_files = generate_plots(data_file)

    # Create the gif
    create_gif(image_files, output_file)

    # Delete the temporary image files
    for image_file in image_files:
        os.remove(image_file)

if __name__ == "__main__":
    main()
