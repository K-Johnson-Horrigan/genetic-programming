# genetic-programming

This fork contains edits made for the final CS824 class project. To run experiments, from the main directory run:
python3 src/models/network/main.py

Outputs will be saved to the saves/ folder.

To run particular experiments, modify the main.py file. For single-objective, set:
'fitness_func': total_interference
For multi-objective, set:
'fitness_func': multi_obj_nsga_esque

To run with a 10x10 network, set: 
'network_shape': (10,10)
To run with a 5x5 network, set: 
'network_shape': (5,5)

To turn on node removal, edit the 'nodes_removed_gen' parameter
