# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import json
import random
import math
from tqdm import tqdm # Library for displaying progress bar

def process_sample(i, snapshot_count, rhod, time, input_params, verbose=False):
    """ Creates a training sample from two points in time. Selects a random output bin for y,
        and saves the output bins for comparison.

    """

    # First sample will always be the first and last element
    if i == 0:
        idxs = [0, snapshot_count-1]
    else:
        # Pick two indexes for snapshots (lowest = input, highest = output)
        idxs = sorted([random.randint(0,snapshot_count-1) for _ in range(2)])
    # Select out the two snapshots
    input_d = rhod[idxs[0]]
    output_d = rhod[idxs[1]]
    if verbose:
        print("indices:", idxs)

    new_input_densities = []
    new_output_densities = []
    # Sum the densities together to aid normalization.
    input_d_sum = np.sum(input_d)
    output_d_sum = np.sum(output_d)
    for i in range(len(input_d)):
        new_input_d = np.sum(input_d[i]) / input_d_sum
        if new_input_d < 1e-30:
            new_input_d = 0.0
        new_input_densities.append(new_input_d)
        
        # Normalize the output bin so we can compare the probability distribution to it
        new_output_d = np.sum(output_d[i]) / output_d_sum
        if new_output_d < 1e-30:
            new_output_d = 0.0
        new_output_densities.append(new_output_d)

    # Time of the input
    t = time[idxs[0]]
        
    # Difference in time (in seconds) between two snapshots
    delta_t = time[idxs[1]] - t
    
    row = np.concatenate([input_params,new_input_densities,[t, delta_t], new_output_densities])
    return row

def write_to_file(data, header=True, batch=False):
    """ Helper method to write training data to a file"""
    columns = ['R', 'Mstar', 'alpha', 'd2g', 'sigma', 'Tgas'] + [f'Input_Bin_{i}' for i in range(bin_count)] + ['t','Delta_t'] + [f'Output_Bin_{i}' for i in range(bin_count)]
    df = pd.DataFrame(res, columns=columns)

    # If writing in batch set the file mode to append
    mode = 'a' if batch else 'w'
    df.to_csv(filename, chunksize=100000, mode=mode, header=header, index=False)

if __name__ == "__main__":

    version = "v2"
    filename = "/scratch/keh4nb/dust_training_data_all_bins_v2.csv"
    root_data_path = f"/project/SDS-capstones-kropko21/dust_models/dust_coag_{version}"

    bin_count = None

    chunk_size = 500
    # Set this to a smaller number to get a smaller training set
    model_count = 10000
    writes = 0
    for d in tqdm(range(model_count)):
        data_set = data_set = str(d).zfill(5)

        data_dir = f"{root_data_path}/data_{data_set}"

        input_params = None
        # Open and extract the input parameters
        with open(os.path.join(root_data_path, f"model_dict_{version}.json")) as f:
            model_dict = json.load(f)
            input_dict = model_dict[data_set]
            input_params = [input_dict['R'], input_dict['Mstar'], input_dict['alpha'],input_dict['d2g'], input_dict['sigma'], input_dict['Tgas']]

        # load model data
        try:
            # `rho_dat`: The dust mass density (in g/cm^3) in each particle size/bin at a given snapshot in time. This is the main "output", i.e., the primary result, of any given model.
            rhod = np.loadtxt(os.path.join(data_dir,"rho_d.dat"))
            # Replace NaNs with 0s
            rhod = np.nan_to_num(rhod)
            # Replace negative values with 0s
            rhod = np.where(rhod<0, 0, rhod) 

            # `a_grid.dat`: The dust particle size in each "bin" in centimeters.
            a_grid = np.loadtxt(os.path.join(data_dir, 'a_grid.dat'))

            # `time.dat`: The time of each snapshot (in seconds).
            time = np.loadtxt(os.path.join(data_dir, "time.dat"))
        except Exception as e:
            print(f'Problem with model {d}; skipped!')
            import traceback
            print(traceback.print_exc())
            continue

        # each line in the density file is one snapshot in time
        # this should match what's in `time.dat`
        snapshot_count = len(rhod)
        # how many size bins are there?
        bin_count = len(a_grid)

        # Set the number of samples
        if snapshot_count > 20:
            # Set the max to 100 for time as 20 cHr 2 is 190
            samples = 190
        else:
            # The number of pairs
            samples = int(math.factorial(snapshot_count) / math.factorial(2) / math.factorial(snapshot_count-2))

        samples += 1
        for i in range(samples):
            row = process_sample(i, snapshot_count, rhod, time, input_params)
            res.append(row)

        # Write to csv every x models to avoid oom
        if d != 0 and d % chunk_size == (model_count - 1) % chunk_size:
            writes += 1
            # Only write the header on first chunk
            header = writes == 1
            write_to_file(res, header, batch=True)
            res = []