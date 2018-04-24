from __future__ import division
import numpy as np
cimport numpy as np
cimport cython

cdef extern from "Python.h":
    ctypedef int Py_intptr_t

cdef extern from "numpy/arrayobject.h":
    ctypedef class numpy.ndarray [object PyArrayObject]:
        cdef char *data
        cdef int nd
        cdef Py_intptr_t *dimensions
        cdef Py_intptr_t *strides
    cdef void import_array()
    cdef int  PyArray_ITEMSIZE(np.ndarray)
    cdef void * PyArray_DATA(np.ndarray)

import_array()

__eps = np.sqrt(np.finfo(np.float64).eps)

def reduction_transfer(
    np.ndarray[dtype=np.uint32_t,  ndim=1, negative_indices=False, mode='c'] ii,
    np.ndarray[dtype=np.uint32_t,  ndim=1, negative_indices=False, mode='c'] j,
    np.ndarray[dtype=np.uint32_t,  ndim=1, negative_indices=False, mode='c'] idx,
    np.ndarray[dtype=np.uint32_t,  ndim=1, negative_indices=False, mode='c'] count,
    np.ndarray[dtype=np.uint32_t,  ndim=1, negative_indices=False, mode='c'] x,
    np.ndarray[dtype=np.float64_t, ndim=1, negative_indices=False, mode='c'] u,
    np.ndarray[dtype=np.float64_t, ndim=1, negative_indices=False, mode='c'] v,
    np.ndarray[dtype=np.float64_t, ndim=1, negative_indices=False, mode='c'] c):
    '''Perform the reduction transfer step from the Jonker-Volgenant algorithm
    
    The data is input in a ragged array in terms of "i" structured as a
    vector of values for each i,j combination where:
    
    ii - the i to be reduced
    j - the j-index of every entry
    idx - the index of the first entry for each i
    count - the number of entries for each i
    x - the assignment of j to i
    u - the dual variable "u" which will be updated. It should be
        initialized to zero for the first reduction transfer.
    v - the dual variable "v" which will be reduced in-place
    c - the cost for each entry.
    
    The code described in the paper is:
    
    for each assigned row i do
    begin
       j1:=x[i]; u=min {c[i,j]-v[j] | j=1..n, j != j1};
       v[j1]:=v[j1]-(u-u[i]);
       u[i] = u;
    end;
    
    The authors note that reduction transfer can be applied in later stages
    of the algorithm but does not seem to provide a substantial benefit
    in speed.
    '''
    cdef:
        int i
        int n_i = ii.shape[0]
        int iii
        int j1
        int j_idx
        int j_temp
        int j_count
        double min_u
        double u_temp
        int j_at_min
        double inf = np.inf
        int *p_i = <int *>(ii.data)
        int *p_j = <int *>(j.data)
        int *p_idx = <int *>(idx.data)
        int *p_count = <int *>(count.data)
        int *p_x = <int *>(x.data)
        double *p_u = <double *>(u.data)
        double *v_base = <double *>(v.data)
        double *c_base = <double *>(c.data)
        double *p_c
    
    with nogil:
        for iii from 0 <= iii < n_i:
            i = p_i[iii]
            min_u = inf
            j1 = p_x[i]
            j_count = p_count[i]
            p_c = c_base + p_idx[i]
            j_at_min = -1
            for j_idx from 0 <= j_idx < j_count:
                j_temp = p_j[j_idx]
                if j_temp != j1:
                    u_temp = p_c[j_idx] - v_base[j_temp]
                    if u_temp < min_u:
                        min_u = u_temp
                        j_at_min = j_temp
            if j_at_min != -1:
                v_base[j1] -= min_u - p_u[i]
                p_u[i] = min_u
        
def augmenting_row_reduction(
    int n,
    np.ndarray[dtype=np.uint32_t,  ndim=1, negative_indices=False, mode='c'] ii,
    np.ndarray[dtype=np.uint32_t,  ndim=1, negative_indices=False, mode='c'] jj,
    np.ndarray[dtype=np.uint32_t,  ndim=1, negative_indices=False, mode='c'] idx,
    np.ndarray[dtype=np.uint32_t,  ndim=1, negative_indices=False, mode='c'] count,
    np.ndarray[dtype=np.uint32_t,  ndim=1, negative_indices=False, mode='c'] x,
    np.ndarray[dtype=np.uint32_t,  ndim=1, negative_indices=False, mode='c'] y,
    np.ndarray[dtype=np.float64_t, ndim=1, negative_indices=False, mode='c'] u,
    np.ndarray[dtype=np.float64_t, ndim=1, negative_indices=False, mode='c'] v,
    np.ndarray[dtype=np.float64_t, ndim=1, negative_indices=False, mode='c'] c):
    '''Perform the augmenting row reduction step from the Jonker-Volgenaut algorithm
    
    n - the number of i and j in the linear assignment problem
    ii - the unassigned i
    jj - the j-index of every entry in c
    idx - the index of the first entry for each i
    count - the number of entries for each i
    x - the assignment of j to i
    y - the assignment of i to j
    u - the dual variable "u" which will be updated. It should be
        initialized to zero for the first reduction transfer.
    v - the dual variable "v" which will be reduced in-place
    c - the cost for each entry.
    
    returns a numpy array of the new free choices.
    '''
    free = np.ascontiguousarray(np.zeros(np.max(y)+1, np.uint32), np.uint32)
    cdef:
        int *p_i = <int *>(ii.data)
        int *p_free = <int *>(PyArray_DATA(free))
        int nfree = 0
        int i
        int j
        int iii
        int jjj
        int n_i = ii.shape[0]
        int n_j
        int *p_j_base     = <int *>(jj.data)
        int *p_idx_base   = <int *>(idx.data)
        int *p_count_base = <int *>(count.data)
        int *p_x_base     = <int *>(x.data)
        int *p_y_base     = <int *>(y.data)
        double *p_u_base  = <double *>(u.data)
        double *p_v_base  = <double *>(v.data)
        double *p_c_base  = <double *>(c.data)
        double *p_c
        double inf = np.inf
        int *p_j
        double u1
        double u2
        double temp
        double eps = __eps
        int i1
        int j1
        int j2
        int k
        
    #######################################
    #
    # From Jonker:
    #
    # procedure AUGMENTING ROW REDUCTION;
    # begin
    # LIST: = {all unassigned rows};
    # for all i in LIST do
    #    repeat
    #    ul:=min {c[i,j]-v[j] for j=l ...n};
    #    select j1 with c [i,j 1] - v[j 1] = u1;
    #    u2:=min {c[i,j]-v[j] for j=l ...n,j< >jl} ;
    #    select j2 with c [i,j2] - v [j2] = u2 and j2 < >j 1 ;
    #    u[i]:=u2;
    #    if ul <u2 then v[jl]:=v[jl]-(u2-ul)
    #    else if jl is assigned then jl : =j2;
    #    k:=y [jl]; if k>0 then x [k]:=0; x[i]:=jl; y [ j l ] : = i ; i:=k
    #  until ul =u2 (* no reduction transfer *) or k=0 i~* augmentation *)
    #  end
    
    with nogil:
        k = 0
        while k < n_i:
            i = p_i[k]
            k += 1
            n_j = p_count_base[i]
            p_j = p_j_base + p_idx_base[i]
            p_c = p_c_base + p_idx_base[i]
            u1 = inf
            u2 = inf
            # Find u1 and u2
            for jjj from 0 <= jjj < n_j:
                j = p_j[jjj]
                temp = p_c[jjj] - p_v_base[j]
                if temp < u1:
                    u2 = u1
                    j2 = j1
                    u1 = temp
                    j1 = j
                elif temp < u2:
                    u2 = temp
                    j2 = j
            # perform the reduction
            i1 = p_y_base[j1]
            if u1 + eps < u2:
               p_v_base[j1] = p_v_base[j1] - u2 + u1
            elif i1 != n:
                j1 = j2
                i1 = p_y_base[j1]
            if i1 != n:
                if u1 + eps < u2:
                    k -= 1
                    p_i[k] = i1
                else:
                    p_free[nfree] = i1
                    nfree += 1
            p_x_base[i] = j1
            p_y_base[j1] = i
    free.resize(nfree)
    return free

def augment(
    int n,
    np.ndarray[dtype=np.uint32_t,  ndim=1, negative_indices=False, mode='c'] ii,
    np.ndarray[dtype=np.uint32_t,  ndim=1, negative_indices=False, mode='c'] jj,
    np.ndarray[dtype=np.uint32_t,  ndim=1, negative_indices=False, mode='c'] idx,
    np.ndarray[dtype=np.uint32_t,  ndim=1, negative_indices=False, mode='c'] count,
    np.ndarray[dtype=np.uint32_t,  ndim=1, negative_indices=False, mode='c'] x,
    np.ndarray[dtype=np.uint32_t,  ndim=1, negative_indices=False, mode='c'] y,
    np.ndarray[dtype=np.float64_t, ndim=1, negative_indices=False, mode='c'] u,
    np.ndarray[dtype=np.float64_t, ndim=1, negative_indices=False, mode='c'] v,
    np.ndarray[dtype=np.float64_t, ndim=1, negative_indices=False, mode='c'] c):
    '''Perform the augmentation step to assign unassigned i and j
    
    n - the # of i and j, also the marker of unassigned x and y
    ii - the unassigned i
    jj - the ragged arrays of j for each i
    idx - the index of the first j for each i
    count - the number of j for each i
    x - the assignments of j for each i
    y - the assignments of i for each j
    u,v - the dual variables
    c - the costs
    '''
    #
    # Holder for c[i,j] - v[j]
    #
    d = np.ascontiguousarray(np.zeros(n, np.float64), np.float64)
    pred = np.ascontiguousarray(np.ones(n, np.uint32))
    #
    # The J that all have the currently considered minimum
    #
    scan = np.ascontiguousarray(np.zeros(n, np.uint32), np.uint32)
    #
    # The J that have been reached by done J
    #
    to_do = np.ascontiguousarray(np.zeros(n, np.uint32), np.uint32)
    #
    # The J that have been processed
    #
    ready = np.ascontiguousarray(np.zeros(n, np.uint32), np.uint32)
    #
    # We need to scan all J for each I to find the ones in the to-do list
    # so we need to maintain an array that keeps track of whether a J is
    # on the to-do list. We do this by starting out at -1 and replacing
    # the ones moved off of "to-do" with "i". Then we can check to see if
    # done[j] == i to see if it is to-do or not
    #
    done = np.ascontiguousarray(-np.ones(n, np.uint32), np.uint32)
    on_to_do = np.ascontiguousarray(-np.ones(n, np.uint32), np.uint32)
    eps = np.finfo(np.float32).eps
    cdef:
        int last, low, up
        int i, iii, i1, k
        int n_i = ii.shape[0]
        int n_j, n_j2
        int j, jjj, j1, jidx
        int found
        int n_to_do
        int n_ready
        int *p_i_base     = <int *>(ii.data)
        int *p_j_base     = <int *>(jj.data)
        int *p_j, *p_j2
        int *p_idx_base   = <int *>(idx.data)
        int *p_count_base = <int *>(count.data)
        int *p_x_base     = <int *>(x.data)
        int *p_y_base     = <int *>(y.data)
        double *p_u_base  = <double *>(u.data)
        double *p_v_base  = <double *>(v.data)
        double *p_c_base  = <double *>(c.data)
        double *p_c
        double *p_d_base  = <double *>(PyArray_DATA(d))
        int *p_pred_base  = <int *>(PyArray_DATA(pred))
        int *p_to_do      = <int *>(PyArray_DATA(to_do))
        int *p_scan       = <int *>(PyArray_DATA(scan))
        int *p_ready      = <int *>(PyArray_DATA(ready))
        int *p_done       = <int *>(PyArray_DATA(done))
        int *p_on_to_do   = <int *>(PyArray_DATA(on_to_do))
        double inf = np.sum(c) + 1 # This is larger than any path through.
        double umin, temp, h, u1
    
    ##################################################
    #
    # Augment procedure: from the Jonker paper.
    #
    # procedure AUGMENT;
    # begin
    #   for all unassigned i* do
    #   begin
    #     for j:= 1 ... n do 
    #       begin d[j] := c[i*,j] - v[j] ; pred[j] := i* end;
    #     READY: = { ) ; SCAN: = { } ; TODO: = { 1 ... n} ;
    #     repeat
    #       if SCAN = { } then
    #       begin
    #         u = min {d[j] for j in TODO} ; 
    #         SCAN: = {j | d[j] = u} ;
    #         TODO: = TODO - SCAN;
    #         for j in SCAN do if y[j]==0 then go to augment
    #       end;
    #       select any j* in SCAN;
    #       i := y[j*]; SCAN := SCAN - {j*} ; READY: = READY + {j*} ;
    #       for all j in TODO do if u + c[i,j] < d[j] then
    #       begin
    #         d[j] := u + c[i,j]; pred[j] := i;
    #         if d[j] = u then
    #           if y[j] is unassigned then go to augment else
    #           begin SCAN: = SCAN + {j} ; TODO: = TODO - {j} end
    #       end
    #    until false; (* repeat always ends with go to augment *)
    #augment:
    #   (* price updating *)
    #   for k in READY do v[k]: = v[k] + d[k] - u;
    #   (* augmentation *)
    #   repeat
    #     i: = pred[j]; y[ j ] := i ; k:=j; j:=x[i]; x[i]:= k
    #   until i = i*
    #  end
    #end
    with nogil:
        for iii from 0 <= iii < n_i:
            i = p_i_base[iii]
            #
            # Initialization
            #
            p_j = p_j_base + p_idx_base[i]
            n_j = p_count_base[i]
            p_c = p_c_base + p_idx_base[i]
            #
            # Initialize d to "inf" for j that are not in the sparse list
            #
            for jjj from 0 <= jjj < n:
                p_d_base[jjj] = inf
            for jjj from 0 <= jjj < n_j:
                j = p_j[jjj]
                p_d_base[j] = p_c[jjj] - p_v_base[j]
                p_to_do[jjj] = j
                p_on_to_do[j] = i
                p_pred_base[j] = i
            
            n_to_do = n_j
            low = 0
            up = 0
            n_ready = 0
            while True:
                if up == low:
                    low = 0
                    up = 0
                    umin = inf
                    #
                    # Find the minimum d[j] among those to do
                    # At the same time, move the j at the current
                    # minimum to the range, low to up, and the remaining
                    # starting at up.
                    #
                    for jjj from 0 <= jjj < n_to_do:
                        j = p_to_do[jjj]
                        if p_done[j] == i:
                            continue
                        temp = p_d_base[j]
                        if temp <= umin:
                            if temp < umin:
                                up = 0
                                umin = temp
                            p_scan[up] = j
                            up += 1
                    j1 = n
                    for jjj from low <= jjj < up:
                        j = p_scan[jjj]
                        if p_y_base[j] == n:
                            # Augment if not assigned
                            j1 = j
                            break
                        p_done[j] = i
                    if j1 < n:
                        break
                #
                # get the first j to be scanned
                #
                # p_j2 points to the j at the i to be replaced
                # n_j2 is the number of j in the sparse array for this i
                #
                j1 = p_scan[low]
                low += 1
                p_ready[n_ready] = j1
                n_ready += 1
                i1 = p_y_base[j1]
                p_j2 = p_j_base + p_idx_base[i1]
                p_c  = p_c_base + p_idx_base[i1]
                n_j2 = p_count_base[i1]
                jidx = bsearch(p_j2, n_j2, j1)
                u1 = p_c[jidx] - p_v_base[j1] - umin
                j1 = n
                for jjj from 0 <= jjj < n_j2:
                    j = p_j2[jjj]
                    if p_done[j] != i:
                        h = p_c[jjj] - p_v_base[j] - u1
                        if h < p_d_base[j]:
                            p_pred_base[j] = i1
                            p_d_base[j] = h
                            if h <= umin:
                                if p_y_base[j] == n:
                                    j1 = j
                                    break
                                #
                                # This one is at the minimum so it can be
                                # added to the j currently being scanned
                                #
                                p_scan[up] = j
                                p_done[j] = i
                                up += 1
                            elif p_on_to_do[j] != i:
                                #
                                # This one is reachable, so it can be added
                                # to the list of j to consider.
                                #
                                p_to_do[n_to_do] = j
                                p_on_to_do[j] = i
                                n_to_do += 1
                            
                if j1 != n:
                    break
                            
            # Augment
            for jjj from 0 <= jjj < n_ready:
                j = p_ready[jjj]
                temp = p_v_base[j]
                p_v_base[j] += p_d_base[j] - umin
            while True:
                i1 = p_pred_base[j1]
                p_y_base[j1] = i1
                j1, p_x_base[i1] = p_x_base[i1], j1
                if i1 == i:
                    break
    #
    # Re-establish slackness since we didn't pay attention to u
    #
    for i from 0 <= i < n:
        j = x[i]
        p_j = p_j_base + p_idx_base[i]
        jidx = bsearch(p_j, p_count_base[i], j)
        u[i] = c[p_idx_base[i] + jidx] - v[j]

#
# A binary search function:
#
# ptr - array to search
# count - # of values in the array
# val - value to search for
#
# returns the position of the value relative to the pointer
#
cdef int bsearch(int *ptr, int count, int val) nogil:
    cdef:
        int low = 0
        int high = count-1
        int mid
    while low <= high:
        mid = (low + high) // 2
        if val == ptr[mid]:
            return mid
        if val > ptr[mid]:
            low = mid + 1
        else:
            high = mid - 1
