import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from torsk.visualize import animate_double_imshow
from torsk import hpopt
from torsk.imed import imed_metric

# hp_paths = hpopt.get_hpopt_dirs(".")
# 
# metrics = []
# for path in hp_paths:
#     with nc.Dataset(path.joinpath("pred_data.nc"), "r") as src:
#         outputs = src["outputs"][:]
#         labels = src["labels"][:]
#         # anim = animate_double_imshow(outputs, labels)
#         # plt.show()
#         m = src["metric"][:]
#         plt.plot(m)
#         metrics.append(m)
# plt.show()


fname = "output/pred_data_idx0.nc"
with nc.Dataset(fname, "r") as src:
    real_pixels = src["labels"][:]
    predicted_pixels = src["outputs"][:]

# fig, ax = plt.subplots(1, 2)
# im = ax[0].imshow((real_pixels > 0.9).sum(axis=0))
# plt.colorbar(im, ax=ax[0])
# im = ax[1].imshow((predicted_pixels > 0.9).sum(axis=0))
# plt.colorbar(im, ax=ax[1])
# plt.show()

plt.plot(imed_metric(real_pixels, predicted_pixels))
error = np.abs(predicted_pixels - real_pixels)
plt.plot(error.sum(axis=-1).sum(axis=-1))
plt.show()

error[0,0,0] = 1.
error[0,1,0] = 0.
anim = animate_double_imshow(real_pixels, predicted_pixels)
plt.show()

