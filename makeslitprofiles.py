import numpy as np
import tensorflow as tf
from tensorflow import keras
import csv
import os
import sys
import pandas as pd
import scipy.constants as const
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from scipy.integrate import simpson
from cmcrameri import cm 
from IPython import display

tf.config.experimental.enable_tensor_float_32_execution(False)

#### PLOTTING ###
num_colors = 20

def lighten_colors(colors, amount=0.5):
    """Lightens RGB colors by blending with white."""
    white = np.ones(3)
    return colors[:, :3] * (1 - amount) + white * amount

# Generate original colors
colours_X = cm.vanimo(np.linspace(0, 0.49, num_colors))
colours_C = cm.vanimo(np.linspace(1, 0.51, num_colors))

# Lighten them
light_colours_X = lighten_colors(colours_X, amount=0.4)
light_colours_C = lighten_colors(colours_C, amount=0.4)

# Repeat as before
color_cycle_X = np.tile(light_colours_X, (int(np.ceil(1000000 / num_colors)), 1))[:1000000]
color_cycle_C = np.tile(light_colours_C, (int(np.ceil(1000000 / num_colors)), 1))[:1000000]

plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_cycle_X)

params = {"axes.labelsize": 14,
          "axes.titlesize": 16,}
plt.rcParams["axes.linewidth"] = 1
plt.rcParams['mathtext.bf'] = 'STIXGeneral:italic:bold'
plt.rcParams['figure.dpi'] = 100
plt.rcParams.update(params)

def place(ax):
  ax.tick_params(direction="in", which="minor", length=3)
  ax.tick_params(direction="in", which="major", length=5, labelsize=13)
  ax.grid(which="major", ls="--", dashes=(1, 3), lw=0.8, zorder=0)
  #ax.legend(frameon=True, loc="best", fontsize=12,edgecolor="black")
  fig.tight_layout()

def configure_plot(zbins):
    fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharex=True)
    ax[0].set_xlim(zbins[0], zbins[-1])
    ax[1].set_ylim(-10, 2)
    ax[0].set_ylabel(r'$\rho(z)$ [$\mathrm{\AA}^{-3}$]')
    ax[1].set_ylabel(r'$\beta[\mu - V_{\mathrm{ext}}(z) - q \phi_\mathrm{R}(z)]$')
    ax[0].set_xlabel(r'$z$ [$\mathrm{\AA}$]')
    ax[1].set_xlabel(r'$z$ [$\mathrm{\AA}$]')
    ax[1].yaxis.set_label_position("right")
    ax[1].yaxis.tick_right()
    
    ax[0].grid(which="major", ls="dashed", dashes=(1, 3), lw=0.5, zorder=0)
    ax[1].grid(which="major", ls="dashed", dashes=(1, 3), lw=0.5, zorder=0)
    ax[0].tick_params(direction="in", which="major", length=5, labelsize=13)
    ax[1].tick_params(direction="in", which="major", length=5, labelsize=13)
    
    plt.tight_layout()
    return fig, ax

def plot_interactive_SR_twotype(fig, ax, zbins, rho_H, rho_O, muloc_H, muloc_O, color_count):
    display.clear_output(wait=True)
    ax[0].plot(zbins, rho_H, color=color_cycle_X[color_count],lw=1)
    ax[0].plot(zbins, rho_O, ls="--", color=color_cycle_C[color_count],lw=1)

    line_H, = ax[1].plot(zbins, muloc_H, color=color_cycle_X[color_count], lw=1, label="N2")
    line_O, = ax[1].plot(zbins, muloc_O, ls="--", color=color_cycle_C[color_count], lw=1, label="CO2")

    ax[1].legend(handles=[line_H, line_O])

    ax[0].set_title(rf"T = {T}, $\mu_X$ = {muX}, $\mu_C$ = {muC}", fontsize = 13, pad=10)
    
    display.display(fig)

def plot_end_SR_twotype(zbins, rhoH, rhoO, mulocH, mulocO, ax):
    display.clear_output(wait=False)
    ax[0].plot(zbins, rhoH, color='black', lw=2)
    ax[0].plot(zbins, rhoO, ls='--', color='black', lw=2)
    line_H, = ax[1].plot(zbins, mulocH, color='black', lw=2, label="N2")
    line_O, = ax[1].plot(zbins, mulocO, ls='--', color='black', lw=2, label="CO2")
    ax[1].legend(handles=[line_H, line_O])

#######


@keras.utils.register_keras_serializable()
class GradientLayer(keras.layers.Layer):
    def call(self, inputs):
        # Compute numerical gradient using central difference (approximated)
        grad = 0.5 * (inputs[:, 2:] - inputs[:, :-2])  # Central difference
        grad = tf.pad(grad, [[0, 0], [1, 1]])  # Pad to keep the same shape
        return grad


def generate_windows(array, bins):
    """
    Generate sliding windows for the input array with a given bin size.

    Parameters:
    - array (np.ndarray): Input array.
    - bins (int): Number of bins on each side of the central bin.
    - mode (str): Padding mode for np.pad (default is "wrap").

    Returns:
    - np.ndarray: Array of sliding windows.
    """
    padded_array = np.pad(array, bins, mode="wrap")
    windows = np.empty((len(array), 2 * bins + 1))
    for i in range(len(array)):
        windows[i] = padded_array[i:i + 2 * bins + 1]
    return windows

def get_c1(model_X, model_C, density_X, density_C, elec, params):
    input_bins = model_X.input_shape[1][1]
    window_bins = (input_bins - 1) // 2
    rhoX_windows = generate_windows(density_X, window_bins).reshape(density_X.shape[0], input_bins, 1)
    rhoC_windows = generate_windows(density_C, window_bins).reshape(density_C.shape[0], input_bins, 1)
    elec_windows = generate_windows(elec, window_bins).reshape(elec.shape[0], input_bins, 1)
    
    paramsInput = {key: tf.convert_to_tensor(np.full(density_X.shape[0], value)) for key, value in params.items()}
    c1X_result = model_X.predict_on_batch({"rho_X": rhoX_windows, "rho_C": rhoC_windows, "elec": elec_windows, **paramsInput}).flatten()
    c1C_result = model_C.predict_on_batch({"rho_X": rhoX_windows, "rho_C": rhoC_windows, "elec": elec_windows, **paramsInput}).flatten()
    return c1X_result, c1C_result 

def betaFexc(modelX, modelC, density_X, density_C, elec, T, dx):
    """
    Calculate the excess free energy Fexc for a given density profile with functional line integration.

    dx: The discretization of the input layer of the model
    """
    alphas = np.linspace(0, 1, 100)
    integrands = np.empty_like(alphas)
    for i, alpha in enumerate(alphas):
        c1X = get_c1(modelX, alpha * density_X, density_C, elec, params={"T": T})
        c1C = get_c1(modelC, density_X, alpha * density_C, elec, params={"T": T})
        
        integrands[i] = np.sum(density_X * c1X + density_C * c1C) * dx
        
    Fexc = -simpson(integrands, x=alphas)
    return Fexc

alpha_updates_default= {
    10: 0.0001,
    20: 0.0005,
    50: 0.001,
    100: 0.005,
    300: 0.005,
    900: 0.008,
    2000: 0.01,
    5000: 0.1,
}

def minimise_SR_twotype(model_H, model_O, zbins, betamuloc_H, betamuloc_O, elec, temp, initial_guess,
                        plot=True, maxiter=100000, alpha_initial=0.00001, 
                        alpha_updates=None,
                        print_every=500, plot_every=500, tolerance=1e-5):
    """
    Solve for two-species density profiles via Picard iteration using neural DFT.

    Parameters:
    - model_H, model_O : tf.keras.Model
        Neural models for species H and O direct correlation functions.
    - zbins : array
        Spatial grid points along z.
    - muloc_H, muloc_O : array
        Local chemical potentials for H and O.
    - elec : array
        Electric potential profile.
    - temp : float
        Temperature (K).
    - plot : bool
        Enable interactive plotting (default: True).
    - maxiter : int
        Max number of iterations (default: 100000).
    - alpha_initial : float
        Initial mixing factor (default: 1e-5).
    - alpha_updates : dict or None
        Optional updates to alpha at specific steps.
    - initial_guess : float
        Initial density guess (default: 0.04).
    - print_every, plot_every : int
        Log/plot frequency (default: 500).
    - tolerance : float
        Convergence threshold (default: 1e-5).

    Returns:
    - zbins, rho_H, rho_O : arrays if converged, else (None, None, None)
    """
    
    # setting up grid
    rho_H_new = np.zeros_like(zbins)
    rho_O_new = np.zeros_like(zbins)
    validH = np.isfinite(betamuloc_H) 
    validO = np.isfinite(betamuloc_O)
    rho_H = initial_guess * np.ones_like(zbins)
    rho_O = initial_guess * np.ones_like(zbins)
    log_rho_H_new = np.zeros_like(zbins)
    log_rho_O_new = np.zeros_like(zbins)
    log_rho_H = np.zeros_like(zbins)
    log_rho_O = np.zeros_like(zbins)
    log_rho_H[validH] = np.log(initial_guess)
    log_rho_O[validO] = np.log(initial_guess)
    log_rho_H[~validH] = -np.inf
    log_rho_O[~validO] = -np.inf
    


    # Picard iteration parameter
    alpha = alpha_initial
    if alpha_updates is None:
        alpha_updates = alpha_updates_default
    
    if plot:
        fig, ax = configure_plot(zbins)
        color_count = 0
  
    for i in range(maxiter + 1):
        if i in alpha_updates:
            alpha = alpha_updates[i]
        
    
        # correlation from trained SR mod
        c1_H_pred, c1_O_pred = get_c1(model_H, model_O, rho_H, rho_O, elec, {"T": temp})
        
        if plot and i % plot_every == 0:
            plot_interactive_SR_twotype(fig, ax, zbins, rho_H, rho_O, betamuloc_H, betamuloc_O, color_count)
            color_count += 1
        # update density
        log_rho_H_new[validH] = betamuloc_H[validH]  + c1_H_pred[validH] - np.log(1/(temp**1.5))
        log_rho_O_new[validO] = betamuloc_O[validO]  + c1_O_pred[validO] - np.log(1/(temp**1.5))
        log_rho_H_new[~validH] = -np.inf
        log_rho_O_new[~validO] = -np.inf
        rho_H_new = np.exp(log_rho_H_new)
        rho_O_new = np.exp(log_rho_O_new)
        log_rho_H = (1 - alpha) * log_rho_H + alpha * log_rho_H_new
        log_rho_O = (1 - alpha) * log_rho_O + alpha * log_rho_O_new
        rho_H = np.exp(log_rho_H)
        rho_O = np.exp(log_rho_O)
        
        delta_H = np.max(np.abs(rho_H_new - rho_H))
        delta_O = np.max(np.abs(rho_O_new - rho_O))
        delta = max(delta_H, delta_O)
        
        if np.isnan(delta):
            print("Not converged: delta is NaN")
            return  None, None, None

        
        if i % print_every == 0:
            print(f"Iteration {i}: delta = {delta}")

        if delta < tolerance:
            print(f"Converged after {i} iterations (delta = {delta})")
            if plot:
                plot_end_SR_twotype(zbins, rho_H, rho_O, betamuloc_H, betamuloc_O, ax)
            return zbins, rho_H, rho_O
        
    print(f"Not converged after {maxiter} iterations (delta = {delta})")
    return None, None, None #zbins, best_rho_H, best_rho_O


def LJ_wall_93(z_range, z_wall, epsilon, sigma, cutoff, place='lo'):
    """
    Calculate the Lennard-Jones wall potential for a range of z values.
 
    epsilon is in kT
    sigma is in Angstrom
    cutoff is in Angstrom
    """
 
    hilo = {'lo': 1.0, 'hi': -1.0}
    z_rel = hilo[place]*(z_range - z_wall)
    
    # Avoid division by zero and negative values in z_rel
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio3 = np.where(z_rel > 0, (sigma / z_rel) ** 3, 0)
        ratio9 = ratio3 ** 3
        energy = epsilon * (ratio9 * 2./15. - ratio3)
        
    # Apply boundary conditions
    energy[z_rel <= 0] = np.inf
    energy[z_rel > cutoff] = 0.0
    
    # Shift the potential to ensure it is zero at the cutoff
    cutoff_energy = epsilon * ((sigma / cutoff) ** 9 * 2./15. - (sigma / cutoff) ** 3)
    energy[z_rel <= cutoff] -= cutoff_energy
    
    return energy

def write_profile(filename, centers, densities_X, densities_C, metadata=None):
    """
    Write the density profiles to a file.

    Parameters:
    - filename (str): Output file name.
    - centers (np.ndarray): Bin centers.
    - densities_H (np.ndarray): Density values for component H.
    - densities_O (np.ndarray): Density values for component O.
    """
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')

        if metadata:
            for key, value in metadata.items():
                writer.writerow([key, value])
            writer.writerow([])

        writer.writerow(["xbins", "rho_X", "rho_C"])
        for center, density_X, density_C in zip(centers, densities_X, densities_C):
            writer.writerow([f"{center:.4f}", f"{density_X:.20f}", f"{density_C:.20f}"])
    

def calculate_slit_profile(muX, muC, epsilon_X, epsilon_C, sigma_X, sigma_C, cutoff, temp, modelX_path, modelC_path, results_path, plot_path, initial_guess,dx, H, plot, vacuum):

    model_X = keras.models.load_model(modelX_path)
    model_C = keras.models.load_model(modelC_path)

    os.makedirs(results_path, exist_ok=True)
    os.makedirs(plot_path, exist_ok=True)
    
    #plotfile = f'{plot_path}/slit_muX{muX}_muC{muC}_epsX{epsilon_X}_epsC{epsilon_C}_sigX_{sigma_X}_sigC{sigma_C}_T{T}_H{H}.png'
    #output_file = f'{results_path}/slit_muX{muX}_muC{muC}_epsX{epsilon_X}_epsC{epsilon_C}_sigX_{sigma_X}_sigC{sigma_C}_{T}_H{H}.out'
    output_file = f"{results_path}/123.out"
    plotfile = f'{plot_path}/123.png'
    
    zbins = np.arange(0, 2*vacuum+H+dx, dx)
    elec = np.zeros_like(zbins)

    beta = 1/(const.k * temp)
    
    # LJ_wall_93(z_range, z_wall, epsilon, sigma, cutoff, place='lo')
    Vext_X = LJ_wall_93(zbins, vacuum, epsilon_X, sigma_X, cutoff, 'lo') + LJ_wall_93(zbins, vacuum+H, epsilon_X, sigma_X, cutoff, 'hi')
    Vext_C = LJ_wall_93(zbins, vacuum, epsilon_C, sigma_C, cutoff, 'lo') + LJ_wall_93(zbins, vacuum+H, epsilon_C, sigma_C, cutoff, 'hi')

    muloc_X = (-Vext_X + muX) * beta
    muloc_C = (-Vext_C + muC) * beta

    metadata = {
        'muX': muX,
        'muC': muC,
        'epsilon_X': epsilon_X,
        'epsilon_C': epsilon_C,
        'sigma_X': sigma_X,
        'sigma_C': sigma_C,
        'cutoff': cutoff,
        'temperature': temp,
        'modelX_path': modelX_path,
        'modelC_path': modelC_path,
        'dx': dx,
        'zmin': zbins[0],
        'zmax': zbins[-1],
        'num_bins': len(zbins),
        'H': H
        'vacuum': vacuum
        'box_size': H + 2 * vacuum
        'initial_guess': initial_guess
    }

    zs, rho_X, rho_C = minimise_SR_twotype(model_X, model_C, zbins, muloc_X, muloc_C, elec, temp, initial_guess=initial_guess,print_every=100, plot = plot)
    if zs is not None:
        write_profile(output_file, zs, rho_X, rho_C, metadata)
        if plot:
            plt.gcf().tight_layout()
            plt.gcf().savefig(plotfile, dpi=300)
    return zs, rho_X, rho_C


################################

results_path = './slit_data'
plot_path = './slit_plots'

modelC_path = "/scratch/fb590/co2-n2/models/c1_C.keras"
modelX_path = "/scratch/fb590/co2-n2/models/c1_X.keras"

dx = 0.02

temperatures = [200, 250, 300, 310, 320, 350, 400]
slit_lengths = [10, 15, 20, 25, 30, 35, 50, 75, 100]

mu_range_CO2_kelvin = [-2500, -2150, -1800, -1450, -1100, -750]
mu_range_CO2_joules = [x * const.k for x in mu_range_CO2_kelvin]

mu_range_N2_kelvin = [-2250, -1900, -1550, -1200, -850, -500]
mu_range_N2_joules = [x * const.k for x in mu_range_N2_kelvin]

epsilon_X = 1000 * const.Avogadro
epsilon_C = 1000 * const.Avogadro

sigma_X = 1.0
sigma_C = 1.0
cutoff = 5.0

vacuum = 10

plot = True

initial_guess = 0.01 

for T in temperatures:
    for H in lengths:
        z_range, rho_X, rho_C = calculate_slit_profile(
            muX, muC,
            epsilon_X, epsilon_C,
            sigma_X, sigma_C, cutoff,
            T,
            modelX_path, modelC_path,
            results_path, plot_path,
            initial_guess,
            dx, H,
            plot, vacuum
        )
