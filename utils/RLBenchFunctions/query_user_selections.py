import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button
import tkinter as tk

import tkinter as tk
import pickle
import os
import numpy as np

import tkinter as tk
import pickle
import os
import numpy as np

def launch_trajectory_selector_from_config(config, save_figures=True):
    # Load corrections
    with open(os.path.join(config.iteration_working_dir, "all_corrections.pkl"), 'rb') as f:
        all_corrections = pickle.load(f)

    num_rows = len(all_corrections)
    #num_cols = len(all_corrections[0].corrections)
    num_cols = 0
    for i in all_corrections:
        num_cols_n = len(i.corrections)
        if num_cols_n > num_cols:
            num_cols = num_cols_n

    state_matrix = np.zeros((num_rows, num_cols), dtype=int)

    root = tk.Tk()
    root.title("Trajectory Selector")

    current_mode = tk.IntVar(value=0)

    grid_frame = tk.Frame(root, padx=5, pady=5)
    grid_frame.pack()

    buttons = [[None for _ in range(num_cols)] for _ in range(num_rows)]

    def get_color(val):
        return {
            -1: "red",
            0: "white",
            1: "gold",
            2: "green"
        }.get(val, "white")

    def on_cell_click(i, j):
        current_val = state_matrix[i, j]
        selected_mode = current_mode.get()

        if selected_mode == 0:
            return

        if current_val == selected_mode:
            state_matrix[i, j] = 0
        else:
            state_matrix[i, j] = selected_mode

        buttons[i][j].configure(bg=get_color(state_matrix[i, j]))

    # Top-left blank
    tk.Label(grid_frame, text="", width=4).grid(row=0, column=0)

    for j in range(num_cols):
        tk.Label(grid_frame, text=f"T{j}", font=("Arial", 7), width=4).grid(row=0, column=j+1, padx=1, pady=1)

    for i in range(num_rows):
        tk.Label(grid_frame, text=f"U{i}", font=("Arial", 7), width=5).grid(row=i+1, column=0, padx=2, pady=1)
        for j in range(num_cols):
            btn = tk.Button(grid_frame, bg="white", width=2, height=1,
                            command=lambda i=i, j=j: on_cell_click(i, j))
            btn.grid(row=i+1, column=j+1, padx=1, pady=1)
            buttons[i][j] = btn

    # Control buttons
    control_frame = tk.Frame(root)
    control_frame.pack(pady=8)

    def set_mode_expert():
        current_mode.set(2)

    def set_mode_semi_expert():
        current_mode.set(1)

    def set_mode_negative_demo():
        current_mode.set(-1)

    def finish_selection():
        # Save the matrix
        save_path = os.path.join(config.iteration_working_dir, "user_preferences.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(state_matrix, f)
        print(f"Saved selection matrix to {save_path}")

        if save_figures:
            fig, ax = plt.subplots(figsize=(num_cols, num_rows))

            for i in range(num_rows):
                for j in range(num_cols):
                    val = state_matrix[i, j]
                    color = get_color(val)
                    rect = Rectangle((j, i), 1, 1, facecolor=color, edgecolor='black', linewidth=1)
                    ax.add_patch(rect)

            ax.set_xlim(0, num_cols)
            ax.set_ylim(0, num_rows)
            ax.set_xticks(np.arange(num_cols) + 0.5)
            ax.set_yticks(np.arange(num_rows) + 0.5)
            ax.set_xticklabels([f"Trajectory {j}" for j in range(num_cols)], rotation=90)
            ax.set_yticklabels([f"Correction {i}" for i in range(num_rows)])
            ax.invert_yaxis()
            ax.set_aspect('equal')
            ax.set_title("User Preference Selection\n(Green=Expert, Gold=Semi-Expert, Red=Negative)", fontsize=13, pad=15)
            plt.tight_layout()

            fig_path = os.path.join(config.iteration_working_dir, "user_preferences_plot.png")
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"Saved preference plot to {fig_path}")
            plt.close()

        root.destroy()

    tk.Button(control_frame, text="Select Expert", command=set_mode_expert,
              bg="lightgreen", font=("Arial", 9)).pack(side="left", padx=8)
    tk.Button(control_frame, text="Select Semi-Expert", command=set_mode_semi_expert,
              bg="khaki", font=("Arial", 9)).pack(side="left", padx=8)
    tk.Button(control_frame, text="Negative Demonstration", command=set_mode_negative_demo,
              bg="lightcoral", font=("Arial", 9)).pack(side="left", padx=8)
    tk.Button(control_frame, text="Done", command=finish_selection,
              bg="lightgray", font=("Arial", 9)).pack(side="left", padx=8)

    root.mainloop()

    return state_matrix


def launch_trajectory_selector_from_config_DEPRECATED(config, save_figures=True):
    # Load corrections
    with open(os.path.join(config.iteration_working_dir, "all_corrections.pkl"), 'rb') as f:
        all_corrections = pickle.load(f)

    num_rows = len(all_corrections)
    num_cols = len(all_corrections[0].corrections)

    state_matrix = np.zeros((num_rows, num_cols), dtype=int)

    root = tk.Tk()
    root.title("Trajectory Selector")

    current_mode = tk.IntVar(value=0)

    grid_frame = tk.Frame(root, padx=5, pady=5)
    grid_frame.pack()

    buttons = [[None for _ in range(num_cols)] for _ in range(num_rows)]

    def get_color(val):
        return {0: "white", 1: "gold", 2: "green"}.get(val, "white")

    def on_cell_click(i, j):
        current_val = state_matrix[i, j]
        selected_mode = current_mode.get()

        if selected_mode == 0:
            return

        if current_val == selected_mode:
            state_matrix[i, j] = 0
        else:
            state_matrix[i, j] = selected_mode

        buttons[i][j].configure(bg=get_color(state_matrix[i, j]))

    # Top-left blank
    tk.Label(grid_frame, text="", width=4).grid(row=0, column=0)

    for j in range(num_cols):
        tk.Label(grid_frame, text=f"T{j}", font=("Arial", 7), width=4).grid(row=0, column=j+1, padx=1, pady=1)

    for i in range(num_rows):
        tk.Label(grid_frame, text=f"U{i}", font=("Arial", 7), width=5).grid(row=i+1, column=0, padx=2, pady=1)
        for j in range(num_cols):
            btn = tk.Button(grid_frame, bg="white", width=2, height=1,
                            command=lambda i=i, j=j: on_cell_click(i, j))
            btn.grid(row=i+1, column=j+1, padx=1, pady=1)
            buttons[i][j] = btn

    # Control buttons
    control_frame = tk.Frame(root)
    control_frame.pack(pady=8)

    def set_mode_expert():
        current_mode.set(2)

    def set_mode_semi_expert():
        current_mode.set(1)

    def finish_selection():
        # Save the matrix
        save_path = os.path.join(config.iteration_working_dir, "user_preferences.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(state_matrix, f)
        print(f"Saved selection matrix to {save_path}")

        if save_figures:
            fig, ax = plt.subplots(figsize=(num_cols, num_rows))

            for i in range(num_rows):
                for j in range(num_cols):
                    val = state_matrix[i, j]
                    color = get_color(val)
                    rect = Rectangle((j, i), 1, 1, facecolor=color, edgecolor='black', linewidth=1)
                    ax.add_patch(rect)

            ax.set_xlim(0, num_cols)
            ax.set_ylim(0, num_rows)
            ax.set_xticks(np.arange(num_cols) + 0.5)
            ax.set_yticks(np.arange(num_rows) + 0.5)
            ax.set_xticklabels([f"Trajectory {j}" for j in range(num_cols)], rotation=90)
            ax.set_yticklabels([f"Correction {i}" for i in range(num_rows)])
            ax.invert_yaxis()
            ax.set_aspect('equal')
            ax.set_title("User Preference Selection (Expert = Green, Semi-Expert = Gold)", fontsize=14, pad=15)
            plt.tight_layout()

            fig_path = os.path.join(config.iteration_working_dir, "user_preferences_plot.png")
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"Saved preference plot to {fig_path}")
            plt.close()

        root.destroy()

    tk.Button(control_frame, text="Select Expert", command=set_mode_expert,
              bg="lightgreen", font=("Arial", 9)).pack(side="left", padx=8)
    tk.Button(control_frame, text="Select Semi-Expert", command=set_mode_semi_expert,
              bg="khaki", font=("Arial", 9)).pack(side="left", padx=8)
    tk.Button(control_frame, text="Done", command=finish_selection,
              bg="lightgray", font=("Arial", 9)).pack(side="left", padx=8)

    root.mainloop()

    return state_matrix

