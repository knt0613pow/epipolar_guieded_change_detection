import json

# with open('I2.json', 'r') as f:
#     data = json.load(f)

# del(data['imageData'])

# with open('I2.json', 'w') as g:
#     json.dump(data, g)
import scipy.sparse as sparse
import numpy as np



sparse_matrix = np.array([[0,1],[0,0]])

np.save('relation.npy', sparse_matrix)
breakpoint()
