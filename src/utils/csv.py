import numpy as np

from src.utils.save import load_kwargs, load_fits

if __name__ == '__main__':

    names = [
        '5x5_mini',
        '10x10_mini',
        '10x10_mini_drop'
    ]
    saves_path = '../../saves/network/'
    num_extra_columns = 1  # Adds extra columns from kwargs but fails with multidimensional kwargs

    for name in names:

        # Load fitness values
        kwargs = load_kwargs(name, saves_path)
        fits = load_fits(**kwargs)

        # Columns for the hyper index for each individual and fitness
        header = ['Test Number', 'Run', 'Generation', 'Individual', 'Fitness']

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

        print(f'Saving {saves_path}{name}/data/{name}.csv')

        np.savetxt(f'{saves_path}{name}/data/{name}.csv', table, delimiter=',', fmt='%s')