import numpy as np;
import scipy as sp;
import skimage.morphology as morf;
import skimage.filters as filt;
import scipy.ndimage as ndi;
from scipy.fftpack import dct,idct,dctn,idctn;
from netCDF4 import Dataset;
from joblib import Parallel, delayed
import multiprocessing;
import sys;

num_cores = multiprocessing.cpu_count()

#(input_file,output_file) = sys.argv;
#(input_file,output_file,chunk_size,lat_per_k1,lon_per_k2) = ("kuro_SSH_3daymean.nc","kuro_SSH_3daymean_dct.nc",64,2,2);
#Command line parameters:
# {input,output}_file: What the name says.
# chunk_size: number of frames to process in memory at a time
# lat_per_k1, lon_per_k2: Store k1-frequencies up to nlat/lat_per_k1, and k2-freqs. up to nlon/lon_per_k2.
(input_file,output_file) = sys.argv[1:3];
(chunk_size,lat_per_k1,lon_per_k2) = (int(a) for a in sys.argv[3:6]);


#def smooth_mask_all(input_file,output_file):
nin    = Dataset(input_file,"r");
in_ssh = nin["SSH"];
(ntime,nlat,nlon) = in_ssh.shape;
(nk1,nk2) = (nlat//lat_per_k1, nlon//lon_per_k2);


print("Interpolating masked data and cosine-transforming using FFT from ",input_file,": ",in_ssh.shape," to ",output_file,": ", (ntime,nk1,nk2));

mask=in_ssh[0].mask.copy()
inner_region=np.zeros(mask.shape,dtype=np.bool)
inner_region[in_ssh[0].data==-1.0] = True;
full_mask = np.logical_or(mask,inner_region)
edges=np.logical_and(morf.binary_dilation(full_mask,selem=morf.disk(3)),
                     np.logical_not(full_mask))

nout= Dataset(output_file,"w");

o_time = nout.createDimension('time',ntime);
o_nlat = nout.createDimension('nlat',nlat);
o_nlon = nout.createDimension('nlon',nlon);
o_nlat = nout.createDimension('nk1',nk1);
o_nlon = nout.createDimension('nk2',nk2);

out_full_mask  = nout.createVariable("full_mask",'u1',("nlat","nlon"));
out_inner_mask = nout.createVariable("inner_mask",'u1',("nlat","nlon"));
out_outer_mask = nout.createVariable("outer_mask",'u1',("nlat","nlon"));
out_edge_mask  = nout.createVariable("edge_mask",'u1',("nlat","nlon"));
out_time       = nout.createVariable("time",np.float32,("time",));

out_outer_mask[:,:] = mask[:,:];
out_full_mask[:,:]  = full_mask[:,:];
out_inner_mask[:,:] = inner_region[:,:];
out_edge_mask[:,:]  = edges[:,:];

#out_time[:] = nin["time"][:];
out_ssh_dct  = nout.createVariable("SSH_DCT",np.float32,("time","nk1","nk2"));

full_mask_plus = morf.binary_dilation(full_mask,selem=morf.disk(3));

def smooth_mask_and_dct(i,frame):
    (ny,nx) = frame.shape;
    mask_data=frame.copy();
    mean    = np.mean(mask_data);
    
    for i in range(ny):
        f  = np.ma.masked_array(mask_data[i],mask=np.logical_not(edges[i]));#full_mask[i]);
        f[0]  = mean;
        f[-1] = mean;
        xs = np.nonzero(f)[0];
        fi = sp.interpolate.interp1d(xs,f.compressed(),kind='linear');
    
        mask_xs = np.nonzero(full_mask[i])[0];        
        mask_data[i,mask_xs] = fi(mask_xs)

    for i in range(nx):
        f  = np.ma.masked_array(mask_data[:,i],mask=np.logical_not(edges[:,i]));#full_mask[i]);
        f[0]  = mean;
        f[-1] = mean;
        xs = np.nonzero(f)[0];
        fi = sp.interpolate.interp1d(xs,f.compressed(),kind='linear');
    
        mask_xs = np.nonzero(full_mask[:,i])[0];        
        mask_data[mask_xs,i] = (mask_data[mask_xs,i]+fi(mask_xs))/2;
        
    for i in range(5):
        mask_data[full_mask] = ndi.gaussian_filter(mask_data,sigma=22-4*i)[full_mask];
    mask_data[full_mask_plus] = ndi.gaussian_filter(mask_data,sigma=1)[full_mask_plus];

    return dctn(mask_data,type=2,shape=(nk1,nk2),norm='ortho',overwrite_x=True);

chunk = np.empty((chunk_size,nk1,nk2));
for start in range(0,ntime,chunk_size):
    end = min(start+chunk_size,ntime);
    print(start,":",end);
    chunk[:end-start] = Parallel(n_jobs=num_cores)(delayed(smooth_mask_and_dct)(i,in_ssh[i]) for i in range(start,end));
    out_ssh_dct[start:end] = chunk[:end-start];

