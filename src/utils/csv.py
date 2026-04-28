import os
import sys
script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(script_directory[:-9])

import numpy as np

from src.utils.save import load_kwargs, load_fits, load_extra_1, load_extra_2

if __name__ == '__main__':

    names = [
        'lex_0',
        'nsga',
        'sum'
    ]
    saves_path = 'big_map_saves/placement/'
    num_extra_columns = 1  # Adds extra columns from kwargs but fails with multidimensional kwargs

    for name in names:

        # Load fitness values
        kwargs = load_kwargs(name, saves_path)
        fits = load_fits(**kwargs)

        # Columns for the hyper index for each individual and fitness
        header = ['Test_Number', 'Run', 'Generation', 'Individual', 'Fitness']

        # Test specific columns
        header += kwargs['test_kwargs'][0][:num_extra_columns]

        table = [header]

        for test_num in range(len(kwargs['test_kwargs'])-1):
            test_name = kwargs['test_kwargs'][test_num+1][0]
            for run_num in range(fits.shape[1]):
                for gen_num in range(fits.shape[2]):
                    for org_num in range(fits.shape[3]):

                        fit = fits[test_num, run_num, gen_num, org_num]

                        # Values for the hyper index for each individual and fitness
                        row = [test_num, run_num, gen_num, org_num, fit]

                        # Values for test specific columns
                        row += kwargs['test_kwargs'][test_num+1][:num_extra_columns]

                        table.append(row)

        print(f'Saving {saves_path}{name}/data/{name}_fitness.csv')

        np.savetxt(f'{saves_path}{name}/data/{name}_fitness.csv', table, delimiter=',', fmt='%s')

    for name in names:

        # Load fitness values
        kwargs = load_kwargs(name, saves_path)
        fits = load_extra_1(**kwargs)

        # Columns for the hyper index for each individual and fitness
        header = ['Test_Number', 'Run', 'Generation', 'Individual', 'Connectivity']

        # Test specific columns
        header += kwargs['test_kwargs'][0][:num_extra_columns]

        table = [header]

        for test_num in range(len(kwargs['test_kwargs'])-1):
            test_name = kwargs['test_kwargs'][test_num+1][0]
            for run_num in range(fits.shape[1]):
                for gen_num in range(fits.shape[2]):
                    for org_num in range(fits.shape[3]):

                        fit = fits[test_num, run_num, gen_num, org_num]

                        # Values for the hyper index for each individual and fitness
                        row = [test_num, run_num, gen_num, org_num, fit]

                        # Values for test specific columns
                        row += kwargs['test_kwargs'][test_num+1][:num_extra_columns]

                        table.append(row)

        print(f'Saving {saves_path}{name}/data/{name}_connectivity.csv')

        np.savetxt(f'{saves_path}{name}/data/{name}_connectivity.csv', table, delimiter=',', fmt='%s')

    for name in names:

        # Load fitness values
        kwargs = load_kwargs(name, saves_path)
        fits = load_extra_2(**kwargs)

        # Columns for the hyper index for each individual and fitness
        header = ['Test_Number', 'Run', 'Generation', 'Individual', 'Coverage']

        # Test specific columns
        header += kwargs['test_kwargs'][0][:num_extra_columns]

        table = [header]

        for test_num in range(len(kwargs['test_kwargs'])-1):
            test_name = kwargs['test_kwargs'][test_num+1][0]
            for run_num in range(fits.shape[1]):
                for gen_num in range(fits.shape[2]):
                    for org_num in range(fits.shape[3]):

                        fit = fits[test_num, run_num, gen_num, org_num]

                        # Values for the hyper index for each individual and fitness
                        row = [test_num, run_num, gen_num, org_num, fit]

                        # Values for test specific columns
                        row += kwargs['test_kwargs'][test_num+1][:num_extra_columns]

                        table.append(row)

        print(f'Saving {saves_path}{name}/data/{name}_coverage.csv')

        np.savetxt(f'{saves_path}{name}/data/{name}_coverage.csv', table, delimiter=',', fmt='%s')
