
This fork contains edits made for the final CS812 class project. To run experiments, from the main directory run: python3 src/models/wmn/main.py

Outputs will be saved to the saves/ folder.

To run particular experiments, modify the main.py file. 

Change 'fitness_func' to the following values to update the fitness function:
- for non-domination of points (nsga): multi_obj_nsga_esque
- for lexicase: lexicase_fitness
- for entropy (from previous work): cov_con_entropy_fitness
- for basic sumation of coverage and connectivity scores: cov_con_sum_fitness
