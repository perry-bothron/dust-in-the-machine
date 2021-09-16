# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import json
import random
import math
from tqdm import tqdm # Library for displaying progress bar

def process_sample(i, snapshot_count, rhod, time, input_params, low_density_cut=1e-30, verbose=False):
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
    # Normalize the densities so the sum over all bins is one.
    input_d_sum = np.sum(input_d)
    output_d_sum = np.sum(output_d)
    for i in range(len(input_d)):
        new_input_d = input_d[i] / input_d_sum
        if new_input_d < low_density_cut:
            new_input_d = 0.0
        new_input_densities.append(new_input_d)
        
        # Normalize the output bin so we can compare the probability distribution to it
        new_output_d = output_d[i] / output_d_sum
        if new_output_d < low_density_cut:
            new_output_d = 0.0
        new_output_densities.append(new_output_d)

    # Time of the input
    t = time[idxs[0]]
        
    # Difference in time (in seconds) between two snapshots
    delta_t = time[idxs[1]] - t
    
    row = np.concatenate([input_params,new_input_densities,[t, delta_t], new_output_densities])
    return row

def write_to_csv_file(data, filename, bin_count, header=True, batch=False):
    """ Helper method to write training data to a file."""
    columns = ['R', 'Mstar', 'alpha', 'd2g', 'sigma', 'Tgas'] + [f'Input_Bin_{i}' for i in range(bin_count)] + ['t','Delta_t'] + [f'Output_Bin_{i}' for i in range(bin_count)]
    df = pd.DataFrame(data, columns=columns)

    # If writing in batch set the file mode to append
    if batch:
        mode = 'a'
    else:
        mode = 'w'
    # `chunksize` in this context is how many rows to write at a time.
    df.to_csv(filename, chunksize=100000, mode=mode, header=header, index=False)

if __name__ == "__main__":

    version = "v3"
    prefix = "/scratch/jpr8yu/sds-capstones-2020/"
    out_filename = prefix + f"/dust_training_data_all_bins_{version}.csv"
    root_data_path = prefix + f"dust_coag_{version}"

    interactive = False

    bin_count = None
    # limit output to csv file to prevent oom errors;
    # write samples from only every `chunk_size`-th model to file.
    chunk_size = 1
    # how many models to process?
    model_count = 10000
    writes = 0

    use_old_csv_file = False # false, at least the first time

    # Open and extract the input parameters from JSON dictionary of models
    with open(os.path.join(root_data_path, f"model_dict_{version}.json")) as f:
        model_dict = json.load(f)

    # limit the number of samples per model via a snapshot limit for space and cost concerns
    # when there are 20 snapshots in time, we get 190 samples/pairs of data points.
    snapshot_limit = 30
    sample_limit = int(math.factorial(snapshot_limit) * 0.5 / math.factorial(snapshot_limit-2))

    if interactive:
        bar = tqdm(range(model_count)[::chunk_size])
    else:
        bar = range(model_count)[::chunk_size]

    for d in bar:
        data_set = str(d).zfill(5)
        data_dir = f"{root_data_path}/data_{data_set}"

        input_dict = model_dict[data_set]
        input_params = [input_dict['R'], input_dict['Mstar'], input_dict['alpha'],input_dict['d2g'], input_dict['sigma'], input_dict['Tgas']]

        # load model data
        try:
            # `rho_dat`: The dust mass density (in g/cm^3) in each particle size/bin at a given snapshot in time. This is the main "output", i.e., the primary result, of any given model.
            rhod = np.loadtxt(os.path.join(data_dir,"rho_d.dat"))
            # Replace NaNs with 0s
            if np.any(np.isnan(rhod)):
                print("Warning: NaNs found in the dust density!")
                rhod = np.nan_to_num(rhod)
            # Replace negative values with 0s
            if np.any(rhod < 0.0):
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
        if snapshot_count > snapshot_limit:
            # Limit the number of samples to reduce cost and size of output
            samples = sample_limit
        else:
            # The number of time snapshot pairs
            samples = int(math.factorial(snapshot_count) * 0.5 / math.factorial(snapshot_count-2))

        samples += 1
        if interactive:
             # include no of samples, snapshots in progress bar
            bar.set_postfix_str(s="{:0d},{:0d}".format(samples,snapshot_count))
        else:
            print(d,"/",model_count, ";", "{:6.3f}%;".format(d/model_count*100.0),samples,";",snapshot_count,flush=True)

        res = [] # Store formatted data for output to csv
        for i in range(samples):
            row = process_sample(i, snapshot_count, rhod, time, input_params)
            res.append(row)

        writes += 1
        # Only write the header on first chunk
        header = (writes == 1)
        write_to_csv_file(res, out_filename, bin_count, header=header, batch=use_old_csv_file)
        use_old_csv_file = True
