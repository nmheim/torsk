import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from torsk.visualize import animate_imshow


aguhlas_3daymeans_npz = "/mnt/data/torsk_experiments/aguhlas_SSH_3daymean_x1050:1550_y700:1000.npz"
data = np.load(aguhlas_3daymeans_npz)
ssh = data["SSH"]

anim = animate_imshow(ssh)
plt.show()
