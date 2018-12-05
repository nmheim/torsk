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
import os;

num_cores = multiprocessing.cpu_count()

(input_filename,output_directory) = sys.argv[1:3];
(chunk_size) = int(sys.argv[3]);

output_base_filename = output_directory+"/"+os.path.basename(os.path.splitext(input_filename)[0]);

#def smooth_mask_all(input_file,output_file):
nin    = Dataset(input_filename,"r");
in_ssh = nin["SSH"];
(ntime,nlat,nlon) = in_ssh.shape;



print("Interpolating masked data and least-squares cosine-transforming ",input_filename,": ",in_ssh.shape," to ",output_base_filename);

mask=in_ssh[0].mask.copy()
inner_region=np.zeros(mask.shape,dtype=np.bool)
inner_region[in_ssh[0].data==-1.0] = True;
full_mask = np.logical_or(mask,inner_region)
edges=np.logical_and(morf.binary_dilation(full_mask,selem=morf.disk(3)),
                     np.logical_not(full_mask))
full_mask_plus = morf.binary_dilation(full_mask,selem=morf.disk(3));


def outputfile(method,nk1,nk2):
    nout= Dataset(output_base_filename+"_"+method+"_"+str(nk1)+"_"+str(nk2)+".nc","w");

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

    out_time[:] = nin["time"][:];
    out_ssh_ct  = nout.createVariable("SSH_CT",np.float32,("time","nk1","nk2"));

    # Fxx_min, Fxx_max, Fkk_min, Fkk_max
    n_ranges    = nout.createDimension("n_ranges",4);
    ranges   = nout.createVariable("ranges",np.float,("n_ranges"));
    
    return nout, out_ssh_ct


def smooth_mask(frame):
    (ny,nx) = frame.shape;
    mask_data=frame.data.copy();
    
    for i in range(ny):
        f  = np.ma.masked_array(mask_data[i].copy(),mask=np.logical_not(edges[i]));#full_mask[i]);
        mean = np.mean(f);
        if(np.all(f.mask)):
            mean = np.mean(frame);
        if(f.mask[0]):
            f[0]  = mean;
        if(f.mask[-1]):
            f[-1] = mean;

        xs = np.nonzero(f)[0];
        fi = sp.interpolate.interp1d(xs,f.compressed(),kind='linear');
    
        mask_xs = np.nonzero(full_mask[i])[0];        
        mask_data[i,mask_xs] = fi(mask_xs)
        
    for i in range(nx):
        f  = np.ma.masked_array(mask_data[:,i].copy(),mask=np.logical_not(edges[:,i]));#full_mask[i]);
        mean = np.mean(f);
        if(np.all(f.mask)):
            mean = np.mean(frame);
        if(f.mask[0]):
            f[0]  = mean;
        if(f.mask[-1]):
            f[-1] = mean;
        xs = np.nonzero(f)[0];
        fi = sp.interpolate.interp1d(xs,f.compressed(),kind='linear');
    
        mask_xs = np.nonzero(full_mask[:,i])[0];        
        mask_data[mask_xs,i] = (mask_data[mask_xs,i]+fi(mask_xs))/2;
        
    for i in range(4):
        mask_data[full_mask] = ndi.gaussian_filter(mask_data,sigma=22-4*i)[full_mask];
    mask_data[full_mask_plus] = ndi.gaussian_filter(mask_data,sigma=2)[full_mask_plus];
    
    return mask_data;

# Least-squares approximation to restricted DCT-III / Inverse DCT-II
def sct_basis(nx,nk):
    xs = np.arange(nx);
    ks = np.arange(nk);
    basis = 2*np.cos(np.pi*(xs[:,None]+0.5)*ks[None,:]/nx);        
    return basis;

def isct(fx,basis):  
    fk,_,_,_ = sp.linalg.lstsq(basis,fx);
    return fk;

def sct(fk,basis):
    fx = np.dot(basis,fk);
    return fx;

# def svdct_basis(nx,nk):
#     basis = sct_basis(nx,nx);
#     U,s,Vh = sp.linalg.svd(basis);
#     return U[:,nk],s[:nk,:nk],Vh[:nk,:]

# def isvdt(fx,s,Vh):
#     fk = np.dot(Vh,fx)/s;
#     return fk;

# def svdt(fk,Uh):

def isct2(Fxx,basis1, basis2):
    Fkx = isct(Fxx.T,basis2);
    Fkk = isct(Fkx.T,basis1);
    return Fkk

def sct2(Fxx,basis1, basis2):
    Fkx = sct(Fxx.T,basis1);
    Fxx = sct(Fkx.T,basis2);
    return Fxx


def idct2(Fxx,nk1,nk2):
    Fkk = dctn(Fxx.T,norm='ortho')[:nk2,:nk1];
    return Fkk;

def dct2(Fkk,nx1,nx2):
    Fkk_padded = np.pad(Fkk,[[0,nx2-Fkk.shape[1]],[0,nx1-Fkk.shape[0]]],'constant');
    Fxx = idctn(Fkk_padded.T,norm='ortho');
    return Fxx;


def smooth_mask_and_ict(frame):
    print("Unmasking")
    Fxx = smooth_mask(frame);
    Fxx_min, Fxx_max = np.amin(Fxx), np.amax(Fxx);

    Fkk_sct = [];
    Fkk_dct = [];

    print("Transforming")    
    for i in range(len(ns)):
        Fkk_sct.append(isct2(Fxx,BASIS1[i],BASIS2[i]));
        Fkk_dct.append(idct2(Fxx,NK1[i],NK2[i]));
        Fkk_sct_min, Fkk_sct_max = np.amin(Fkk_sct[i]), np.amax(Fkk_sct[i]);
        Fkk_dct_min, Fkk_dct_max = np.amin(Fkk_dct[i]), np.amax(Fkk_dct[i]);            
    
    return Fkk_sct,Fkk_dct, np.array([Fxx_min,Fxx_max,Fkk_sct_min,Fkk_sct_max,Fkk_dct_min,Fkk_dct_max]);

ns  = [2,3,4,6,8,10];
NK1 = [nlat//n for n in ns];
NK2 = [nlon//n for n in ns];
BASIS1 = [sct_basis(nlat,nk1) for nk1 in NK1];
BASIS2 = [sct_basis(nlon,nk2) for nk2 in NK2];

sct_outputs = [outputfile("SCT",NK1[i],NK2[i]) for i in range(len(ns))];
dct_outputs = [outputfile("DCT",NK1[i],NK2[i]) for i in range(len(ns))];

Fxx_min, Fkk_sct_min, Fkk_dct_min =  np.Inf,  np.Inf,  np.Inf;
Fxx_max, Fkk_sct_max, Fkk_dct_max = -np.Inf, -np.Inf, -np.Inf;


for start in range(0,ntime,chunk_size):
    end = min(start+chunk_size,ntime);
    print(start,":",end);
    chunk  = Parallel(n_jobs=num_cores)(delayed(smooth_mask_and_ict)(in_ssh[i]) for i in range(start,end));

    Fkk_sct = [np.array([c[0][i] for c in chunk]) for i in range(len(ns))];
    Fkk_dct = [np.array([c[1][i] for c in chunk]) for i in range(len(ns))];
    ranges  = np.array([c[2] for c in chunk]);

    Fxx_min     = min(Fxx_min,np.amin(ranges[:,0]))
    Fxx_max     = max(Fxx_max,np.amax(ranges[:,1]))
    Fkk_sct_min = min(Fkk_sct_min,np.amin(ranges[:,2]))
    Fkk_sct_max = max(Fkk_sct_max,np.amax(ranges[:,3]))    
    Fkk_dct_min = min(Fkk_dct_min,np.amin(ranges[:,4]))
    Fkk_dct_max = max(Fkk_dct_max,np.amax(ranges[:,5]))    

    for i in range(len(ns)):
        print("Writing SCT_"+str(NK1[i])+"x"+str(NK2[i]));
        sct_outputs[i][1][start:end] = Fkk_sct[i][:end-start];
        sct_outputs[i][0]["ranges"][:] = [Fxx_min,Fxx_max,Fkk_sct_min,Fkk_sct_max];
        
        print("Writing DCT_"+str(NK1[i])+"x"+str(NK2[i]));
        dct_outputs[i][1][start:end] = Fkk_dct[i][:end-start];        
        dct_outputs[i][0]["ranges"][:] = [Fxx_min,Fxx_max,Fkk_dct_min,Fkk_dct_max];

