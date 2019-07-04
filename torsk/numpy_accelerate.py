import numpy as np
from torsk.config import *

Id = lambda A: A
def void():
    return

if numpy_acceleration == "bohrium":
    import bohrium 

    bh = bohrium
    bh_type  = bh.ndarray
    bh_check = bh.bhary.check    
    to_bh    = bh.array
    bh_flush = bh.flush
    bh_diag  = bh.diag
    bh_dot   = bh.dot
    def to_np(A):
        if bh_check(A):
            return A.copy2numpy()
        else:
            return A

    accel_capabilities = ["streaming"]
        
elif numpy_acceleration == "bh107":
    import bh107

    bh       = bh107
    bh_type  = bh.BhArray
    bh_check = lambda A: isinstance(A,bh.BhArray)    
    to_bh    = bh.BhArray.from_numpy
    bh_flush = bh.flush
    bh_diag  = lambda A: to_bh(np.diag(A.asnumpy()))
    bh_dot   = lambda A,B: to_bh(np.dot(to_np(A), to_np(B)))
    def to_np(A):
        if isinstance(A,bh_type):
            bh.flush()          # Make sure everything pending is already written to A
            return A.asnumpy()
        else:
            return A
    accel_capabilities = ["streaming"]
    
else:
    bh = np
    bh_type  = bh.ndarray
    bh_check = lambda A: isinstance(A,bh_type);
    to_bh      = Id
    to_np      = Id
    bh_flush   = void
    bh_diag    = np.diag
    bh_dot     = np.dot
    accel_capabilities = []

    
    
def before_storage(model):
    numpyize(model)

def after_storage(model,old_state=None):
    bohriumize(model,old_state)        

def bohriumize(model,old_state=None):
    print("Bohriumizing...")
    if old_state is None:
        model.esn_cell.weight_hh.values  = to_bh(model.esn_cell.weight_hh.values)
        model.esn_cell.weight_hh.col_idx = to_bh(model.esn_cell.weight_hh.col_idx)
        model.ones = to_bh(model.ones)
        model.wout = to_bh(model.wout)
    else:
        model.esn_cell.weight_hh.values  = old_state["esn_cell.weight_hh.values"]
        model.esn_cell.weight_hh.col_idx = old_state["esn_cell.weight_hh.col_idx"]
        model.ones = old_state["ones"] 
        model.wout = old_state["wout"]
    print("Done bohriumizing...")

def numpyize(model):
    print("Numpyizing...")
    old_state = {};
    old_state["esn_cell.weight_hh.values"]  = model.esn_cell.weight_hh.values;
    old_state["esn_cell.weight_hh.col_idx"] = model.esn_cell.weight_hh.col_idx;
    old_state["ones"] = model.ones; 
    old_state["wout"] = model.wout;
    
    model.esn_cell.weight_hh.values  = to_np(model.esn_cell.weight_hh.values)
    model.esn_cell.weight_hh.col_idx = to_np(model.esn_cell.weight_hh.col_idx)
    model.ones = to_np(model.ones)
    model.wout = to_np(model.wout)
    
    #E = {}
    for D in [model.__dict__, model.esn_cell.__dict__, model.esn_cell.weight_hh.__dict__]:
        for k,v in D.items():
            if isinstance(v,bh_type):
                print("Numpyizing ",k)
                #E[k] = v
                D[k] = to_np(v)                        
    #return E

    return old_state
