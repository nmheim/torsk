import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt


world_ssh_nc = "/mnt/data/torsk_experiments/world_ctrl_SSH_new.nc"

ystart = 700
yend   = 1000
xstart = 1050
xend   = 1550

ssh_fname = f"/mnt/data/torsk_experiments/aguhlas_SSH_3daymean_x{xstart}:{xend}_y{ystart}:{yend}.npz"

print(f"Loading... y({ystart}:{yend}) x({xstart}:{xend})")
with nc.Dataset(world_ssh_nc, "r") as src:
    ssh = src["SSH"][:, ystart:yend, xstart:xend]
    ssh, mask = ssh.data, ssh.mask
    mask = np.logical_or(mask, mask == -1.)
    ssh[mask] = 0.

print(f"Saving to {ssh_fname}")
np.savez(ssh_fname, SSH=ssh, mask=mask)
