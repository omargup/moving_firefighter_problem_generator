########################################
# Script to generate the Dataset       #
########################################
import movingfp.data as data

# Generate and save instances as yaml files

# Erdos instances
num_graphs =  5 # Number of graphs for each config.
n_list = [20, 50]
pn_list = [2.5, 3.0]
bn_list = [1]
dim_list =  [2]
connected = False

data.erdos_data(n_list=n_list, pn_list=pn_list, dim_list=dim_list, bn_list=bn_list,
                num_graphs=num_graphs, connected=connected, file_path=None)

#if __name__ == "__main__":
