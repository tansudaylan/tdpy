import tdpy
import numpy as np

data = np.random.uniform(0, 1, (100, 5))

param_labels = ["1", "2", "3", "4", "5"]

# labels of the populations
listlablpopl = ['stable']
# number of populations
numbpopl = len(listlablpopl)

# number of features
numbfeat = len(param_labels)


# labels of the parameters
listlablpara = []
for k in range(numbfeat):
    listlablpara.append([param_labels[k], ''])

    
tdpy.plot_grid( 
               np.array(listlablpara), data, listlablpopl=listlablpopl, typeplottdim='hist', 
               pathbase=f"fig")


