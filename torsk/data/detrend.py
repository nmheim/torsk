import numpy as np
import scipy.linalg as la
from torsk.data.utils import upscale, downscale

def polynomial_trend(ft,d):
    """Finds best least-squares 'd'-degree polynomial fitting the series 'ft'."""
    n = len(ft)
    xs = np.arange(n);
    B  = np.array([xs**j for j in range(d+1)]).T
    return la.lstsq(B, ft.reshape(n,1))[0];
    
def cycles(ft,cycle_length):
    """Separetes the series 'ft' into a list cycles of length 'cycle_length'."""
    n = len(ft);
    n_cycles  = n//cycle_length;
    nC = n_cycles*cycle_length;
    return ft[:nC].reshape(n_cycles,cycle_length), ft[nC:];

def separate_trend_scaled(ft,nT,Cycle_length):
    """Separate out the quadratic trend and average cycle from a time series 'ft'.

    Given a time scale nT for which the cycle length is an integer 'Cycle_length', 
    this computes and removes the quadratic trend, then scales 'ft' smoothly to 'nT' 
    points, computes the average cycle, and removes it from the data, and scales 
    back to the original time scale.

    Returns: (ft_detrended,b,C), where b=(b0,b1,b2) are the coefficients of the 
             quadratic trend, and C is the average cycle.

    See also: recombine_trend_scaled(), the inverse of this operation.
    """

    nt = len(ft);
    
    # Extract the quadratic trend
    b=polynomial_trend(ft,2);
    
    # Remove trend
    ts=np.arange(nt);
    trend=b[0]+b[1]*ts+b[2]*ts*ts;
    
    # Remove average cycle
    fT=upscale(ft-trend,nT);
    fC,fr=cycles(fT,Cycle_length);
    C =np.mean(fC,axis=0);
        
    fT_detrended=np.concatenate([
        (fC-C).reshape((-1,)),
        fr-C[:len(fr)]
    ]);

    ft_detrended = downscale(fT_detrended,nt)
    
    return (ft_detrended,b,C);

def recombine_trend_scaled(ft,b,C,nT):
    """Recombine the quadratic trend and average cycle with a de-trended time series.
    
    Given the output of separate_trend_scaled() together with the time scale 'nT',
    this re-combines them into the original time series.
    """    
    nt=len(ft);
    Cycle_length=len(C);
    
    # Add back quadratic trend
    ts=np.arange(nt);
    trend=b[0]+b[1]*ts+b[2]*ts*ts;    
    
    # Add back average cycle 
    fT=upscale(ft+trend,nT);
    fC,fr=cycles(fT,Cycle_length);    
    
    fT_retrended=np.concatenate([
        (fC+C).reshape((-1,)),
        fr+C[:len(fr)]
    ]);
    
    ft_retrended=downscale(fT_retrended,nt);
    
    return ft_retrended;


def separate_trends(Ftkk,nT,Cycle_length):
    (nt,nk1,nk2)=Ftkk.shape;
    
    bkk=np.empty((nk1,nk2,3));
    Ckk=np.empty((nk1,nk2,Cycle_length));
    Ftkk_detrended=np.empty(Ftkk.shape);

    # TODO: use vectorized numpy
    for k1 in range(nk1):
        for k2 in range(nk2):
            (ft,b,C) = separate_trend_scaled(Ftkk[:,k1,k2],nT,Cycle_length);
            Ckk[k1,k2]=C;
            bkk[k1,k2]=b.flatten();
            Ftkk_detrended[:,k1,k2]=ft;
    return (Ftkk_detrended,bkk,Ckk)


def recombine_trends(Ftkk_detrended,bkk,Ckk,nT,Cycle_length):
    (nt,nk1,nk2)=Ftkk_detrended.shape;
    Ftkk=np.empty((nt,nk1,nk2));
    
    #TODO: use vectorized numpy
    for k1 in range(nk1):
        for k2 in range(nk2):
            Ftkk[:,k1,k2] = recombine_trend_scaled(
                Ftkk_detrended[:,k1,k2],bkk[k1,k2],Ckk[k1,k2],nT,Cycle_length
            )
    return Ftkk;
