# import os

# os.environ["OMP_NUM_THREADS"] = "1"

import torch
import emcee
import argparse
import pandas as pd
from scipy.spatial import KDTree
# from multiprocessing import Pool

# Modify the code to set the device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def interpolated_mag(theta, mag_table):
    Teff, log_g, Z, d = theta

    # Move relevant data to CPU for KDTree processing
    kdtree = KDTree(mag_table[['Teff', 'log_g', 'Z']])
    dist, points = kdtree.query(theta[:3], 1)
    
    # Move the result back to GPU if necessary
    mag = torch.tensor(mag_table.iloc[points][['F115W', 'F150W', 'F277W', 'F444W']].values, device=device)
    
    # Convert fluxes to magnitudes, mag_table is AB magnitude at 10 pc
    Mag = mag + 5 * torch.log10(torch.tensor(d) / 10)
    
    return Mag

def log_likelihood(theta, mag_table, y, yerr):
    model_m = interpolated_mag(theta, mag_table)
    return -0.5 * torch.sum(((y - model_m) / yerr) ** 2)

def log_prior(theta):
    Teff, log_g, Z, d = theta
    if 400 < Teff < 2400 and 2.9 < log_g < 5.6 and 0.4 < Z < 1.6 and 9 < d < 4000:
        return torch.tensor(0.0, device=device)
    return torch.tensor(-float('inf'), device=device)

def log_probability(theta, mag_table, y, yerr):
    lp = log_prior(theta)
    if not torch.isfinite(lp):  # Ensure prior is finite
        return -float('inf')
    return (lp + log_likelihood(theta, mag_table, y, yerr)).item()

# Ensure that your tensors are on the correct device
theta = torch.tensor([1000, 4, 1, 1500], device=device)
coefficients = torch.tensor([500, 3, 1, 3000], device=device)

# Set up argument parser
parser = argparse.ArgumentParser(description='Run a Python script with SLURM.')
parser.add_argument('n', type=int, help='Set the value of n.')

# Parse the arguments
args = parser.parse_args()

# Set n based on the command-line argument
n = args.n

mag_table = pd.read_feather('interpolated_ABmag.feather')
observation = pd.read_csv('final.csv')

# Convert observation data to PyTorch tensors
magnitudes = torch.tensor(observation[['MAG_AUTO_F115W', 'MAG_AUTO_F150W', 'MAG_AUTO_F277W', 'MAG_AUTO_F444W']].values, device=device)
magerr = torch.tensor(observation[['MAGERR_AUTO_F115W', 'MAGERR_AUTO_F150W', 'MAGERR_AUTO_F277W', 'MAGERR_AUTO_F444W']].values, device=device)
Mobs, Merr = lambda x: magnitudes[x-1], lambda x: magerr[x-1]

# Set the initial guess and coefficients as PyTorch tensors
theta = torch.tensor([1000, 4, 1, 1500], device=device)
coefficients = torch.tensor([500, 3, 1, 2000], device=device)

# Setup walkers
pos = theta + coefficients * torch.randn((160, 4), device=device)
nwalkers, ndim = pos.shape

# Convert the walker positions to numpy for use with emcee
pos_numpy = pos.cpu().numpy()

# Run the MCMC fitting process
sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability, args=(mag_table, Mobs(n), Merr(n))
)
sampler.run_mcmc(pos_numpy, 20000, progress=True)

# with Pool() as pool:
#     sampler = emcee.EnsembleSampler(
#         nwalkers, ndim, log_probability, pool=pool, args=(mag_table, Mobs(n), Merr(n))
#     ) 
#     sampler.run_mcmc(pos, 20000, progress=True)

flat_samples = sampler.get_chain(discard=6000, thin=10, flat=True)

data = pd.DataFrame(flat_samples, columns=['Teff', 'log_g', '[M/H]', 'd'])
if n<9:
    data.to_csv(f'results/BD0{n}_mcmc.csv')
else:
    data.to_csv(f'results/BD{n}_mcmc.csv')