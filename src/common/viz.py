import matplotlib.pyplot as plt
import numpy as np
import os

def _viz_change_of_theta_mode(Thetas, mode, save_dir=None):
    T = len(Thetas)

    diffs = []
    for t in range(1,T):
        diff = np.linalg.norm(Thetas[t][mode] - Thetas[t-1][mode], ord='fro') / np.linalg.norm(Thetas[t][mode], ord='fro')
        diffs.append(diff)
    
    plt.figure()
    plt.plot(range(T-1), diffs, marker='o')
    plt.xlabel("Time")
    plt.ylabel(f"Change in Theta (Mode {mode})")
    plt.grid()
    if save_dir is not None:
        plt.savefig(f"{save_dir}/change_in_theta_mode_{mode}.png")
    else:
        plt.show()
    plt.close()

def viz_change_of_theta(Thetas, save_dir=None):
    if save_dir is not None:
        save_dir = os.path.join(save_dir, "change")
        os.makedirs(save_dir, exist_ok=True)
    M = len(Thetas[0])
    for m in range(M):
        _viz_change_of_theta_mode(Thetas, m, save_dir=save_dir)

def viz_change_of_theta_flatten(Thetas, save_dir=None):
    if save_dir is not None:
        save_dir = os.path.join(save_dir, "change")
        os.makedirs(save_dir, exist_ok=True)
    T = len(Thetas)
    diffs = []
    for t in range(1,T):
        diff = np.linalg.norm(Thetas[t] - Thetas[t-1], ord='fro') / np.linalg.norm(Thetas[t], ord='fro')
        diffs.append(diff)
    
    plt.figure()
    plt.plot(range(T-1), diffs, marker='o')
    plt.xlabel("Time")
    plt.ylabel("Change in Flattened Theta")
    plt.grid()
    if save_dir is not None:
        plt.savefig(f"{save_dir}/change_in_theta_flatten.png")
    else:
        plt.show()
    plt.close()

def _viz_error_of_theta_mode(Thetas_hat, Thetas_true, mode, save_dir=None):
    T = len(Thetas_hat)

    errors = []
    for t in range(T):
        error = np.linalg.norm(Thetas_hat[t][mode] - Thetas_true[t][mode], ord='fro') / np.linalg.norm(Thetas_true[t][mode], ord='fro')
        errors.append(error)
    
    plt.figure()
    plt.plot(range(T), errors, marker='o')
    plt.xlabel("Time")
    plt.ylabel(f"Relative Error in Theta (Mode {mode})")
    plt.grid()
    if save_dir is not None:
        plt.savefig(f"{save_dir}/error_in_theta_mode_{mode}.png")
    else:
        plt.show()
    plt.close()

def viz_error_of_theta(Thetas_hat, Thetas_true, save_dir=None):
    if save_dir is not None:
        save_dir = os.path.join(save_dir, "error")
        os.makedirs(save_dir, exist_ok=True)
    M = len(Thetas_hat[0])
    for m in range(M):
        _viz_error_of_theta_mode(Thetas_hat, Thetas_true, m, save_dir=save_dir)

def viz_error_of_theta_flatten(Thetas_hat, Thetas_true, save_dir=None):
    if save_dir is not None:
        save_dir = os.path.join(save_dir, "error")
        os.makedirs(save_dir, exist_ok=True)
    T = len(Thetas_hat)
    errors = []
    for t in range(T):
        error = np.linalg.norm(Thetas_hat[t] - Thetas_true[t], ord='fro') / np.linalg.norm(Thetas_true[t], ord='fro')
        errors.append(error)
    
    plt.figure()
    plt.plot(range(T), errors, marker='o')
    plt.xlabel("Time")
    plt.ylabel("Relative Error in Flattened Theta")
    plt.grid()
    if save_dir is not None:
        plt.savefig(f"{save_dir}/error_in_theta_flatten.png")
    else:
        plt.show()
    plt.close()