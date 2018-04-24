from __future__ import absolute_import
from __future__ import print_function
import numpy as np

from ._lapjv import reduction_transfer
from ._lapjv import augmenting_row_reduction
from ._lapjv import augment
from six.moves import range

def lapjv(i, j, costs, wants_dual_variables = False, augmenting_row_reductions = 2):
    '''Sparse linear assignment solution using Jonker-Volgenant algorithm
    
    i,j - similarly-sized vectors that pair the object at index i[n] with
          the object at index j[j]
          
    costs - a vector of similar size to i and j that is the cost of pairing
            i[n] with j[n].
            
    wants_dual_variables - the dual problem reduces the costs using two
            vectors, u[i] and v[j] where the solution is the maximum value of
            np.sum(u) + np.sum(v) where cost[i,j] - u[i] - v[j] >=  0.
            Set wants_dual_variables to True to have u and v returned in
            addition to the assignments.
            
    augmenting_row_reductions - the authors suggest that augmenting row reduction
            be performed twice to optimize the u and v before the augmenting
            stage. The caller can choose a different number of reductions
            by supplying a different value here.

    All costs not appearing in i,j are taken as infinite. Each i in the range,
    0 to max(i) must appear at least once and similarly for j.
    
    returns (x, y), the pairs of assignments that represent the solution
    or (x, y, u, v) if the dual variables are requested.
    '''
    import os
    i = np.atleast_1d(i).astype(int)
    j = np.atleast_1d(j).astype(int)
    costs = np.atleast_1d(costs)
    
    assert len(i) == len(j), "i and j must be the same length"
    assert len(i) == len(costs), "costs must be the same length as i"

    #
    # Find the number of i with non-infinite cost for each j
    #
    j_count = np.bincount(j)
    assert not np.any(j_count == 0), "all j must be paired with at least one i"
    #
    # if you order anything by j, this is an index to the minimum for each j
    #
    j_index = np.hstack([[0], np.cumsum(j_count[:-1])])
    #
    # Likewise for i
    #
    i_count = np.bincount(i)
    assert not np.any(i_count == 0), "all i must be paired with at least one j"
    i_index = np.hstack([[0], np.cumsum(i_count[:-1])])
    
    n = len(j_count) # dimension of the square cost matrix
    assert n == len(i_count), "There must be the same number of unique i and j"
    
    # # # # # # # #
    #
    # Variable initialization:
    #
    # The output variables:
    #
    # x - for each i, the assigned j. -1 indicates uninitialized
    # y - for each j, the assigned i
    # u, v - the dual variables
    #
    # A value of x = n or y = n means "unassigned"
    #
    x = np.ascontiguousarray(np.ones(n, np.uint32) * n)
    y = np.ascontiguousarray(np.ones(n, np.uint32) * n, np.uint32)
    u = np.ascontiguousarray(np.zeros(n, np.float64))
    
    # # # # # # # #
    #
    # Column reduction
    #
    # # # # # # # #
    #
    # For a given j, find the i with the minimum cost.
    #
    order = np.lexsort((-i, costs, j))
    min_idx = order[j_index]
    min_i = i[min_idx]
    #
    # v[j] is assigned to the minimum cost over all i
    #
    v = np.ascontiguousarray(costs[min_idx], np.float64)
    #
    # Find the last j for which i was min_i.
    #
    x[min_i] = np.arange(n).astype(np.uint32)
    y[x[x != n]] = np.arange(n).astype(np.uint32)[x != n]
    #
    # Three cases for i:
    #
    # i is not the minimum of any j - i goes on free list
    # i is the minimum of one j - v[j] remains the same and y[x[j]] = i
    # i is the minimum of more than one j, perform reduction transfer
    #
    assignment_count = np.bincount(min_i[min_i != n])
    assignment_count = np.hstack(
        (assignment_count, np.zeros(n - len(assignment_count), int)))
    free_i = assignment_count == 0
    one_i = assignment_count == 1
    #order = np.lexsort((costs, i)) Replace with this after all is done
    order = np.lexsort((j,i))
    j = np.ascontiguousarray(j[order], np.uint32)
    costs = np.ascontiguousarray(costs[order], np.float64)
    i_index = np.ascontiguousarray(i_index, np.uint32)
    i_count = np.ascontiguousarray(i_count, np.uint32)
    if np.any(one_i): 
        reduction_transfer(
            np.ascontiguousarray(np.argwhere(one_i).flatten(), np.uint32),
            j, i_index, i_count, x, u, v, costs)
    #
    # Perform augmenting row reduction on unassigned i
    #
    ii = np.ascontiguousarray(np.argwhere(free_i).flatten(), np.uint32)
    if len(ii) > 0:
        for iii in range (augmenting_row_reductions):
            ii = augmenting_row_reduction(
                n, ii, j, i_index, i_count, x, y, u, v, costs)
    augment(n, ii,
            j, i_index, i_count, x, y, u, v, costs)
    if wants_dual_variables:
        return x,y,u,v
    else:
        return x,y
    
def slow_reduction_transfer(ii, j, idx, count, x, u, v, c):
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
    for i in ii:
        j1 = x[i]
        jj = j[idx[i]:(idx[i]+count[i])]
        uu = np.min((c[idx[i]:(idx[i]+count[i])] - v[jj])[jj != j1])
        v[j1] = v[j1] - uu + u[i]
        u[i] = uu
        
def slow_augmenting_row_reduction(n, ii, jj, idx, count, x, y, u, v, c):
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
    
    returns the new unassigned i
    '''
        
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
    ii = list(ii)
    k = 0
    limit = len(ii)
    free = []
    while k < limit:
        i = ii[k]
        k += 1
        j = jj[idx[i]:(idx[i] + count[i])]
        uu = c[idx[i]:(idx[i] + count[i])] - v[j]
        order = np.lexsort([uu])
        u1, u2 = uu[order[:2]]
        j1,j2 = j[order[:2]]
        i1 = y[j1]
        if u1 < u2:
            v[j1] = v[j1] - u2 + u1
        elif i1 != n:
            j1 = j2
            i1 = y[j1]
        if i1 != n:
            if u1 < u2:
                k -= 1
                ii[k] = i1
            else:
                free.append(i1)
        x[i] = j1
        y[j1] = i
    return np.array(free,np.uint32)

def slow_augment(n, ii, jj, idx, count, x, y, u, v, c):
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
    
    ##################################################
    #
    # Augment procedure: from the Jonker paper.
    #
    # Note:
    #    cred [i,j] = c [i,j] - u [i] - v[j]
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
    #       for all j in TODO do if u + cred[i,j] < d[j] then
    #       begin
    #         d[j] := u + cred[i,j]; pred[j] := i;
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
    inf = np.sum(c) + 1
    d = np.zeros(n)
    cc = np.zeros((n,n))
    cc[:,:] = inf
    for i in range(n):
        cc[i,jj[idx[i]:(idx[i]+count[i])]] = c[idx[i]:(idx[i]+count[i])]
    c = cc
    for i in ii:
        print("Processing i=%d" % i)
        j = jj[idx[i]:(idx[i] + count[i])]
        d = c[i,:] - v
        pred = np.ones(n, int) * i
        on_deck = []
        ready = []
        scan = []
        to_do = list(range(n))
        try:
            while True:
                print("Evaluating i=%d, n_scan = %d" % (i, len(scan)))
                if len(scan) == 0:
                    ready += on_deck
                    on_deck = []
                    umin = np.min([d[jjj] for jjj in to_do])
                    print("umin = %f" % umin)
                    scan = [jjj for jjj in to_do if d[jjj] == umin]
                    to_do = [jjj for jjj in to_do if d[jjj] != umin]
                    for j1 in scan:
                        if y[j1] == n:
                            raise StopIteration()
                j1 = scan[0]
                iii = y[j1]
                print("Consider replacing i=%d, j=%d" % (iii, j1))
                scan = scan[1:]
                on_deck += [j1]
                u1 = c[iii, j1] - v[j1] - umin
                for j1 in list(to_do):
                    h = c[iii, j1] - v[j1] - u1
                    print("Consider j=%d as replacement, c[%d,%d]=%f,v[%d]=%f,h=%f, d[j]= %f" % (j1,iii,j1,c[iii,j1],j1,v[j1],h,d[j1]))
                    if h < d[j1]:
                        print("Add to chain")
                        pred[j1] = iii
                        if h == umin:
                            if y[j1] == n:
                                raise StopIteration()
                            print("Add to scan")
                            scan += [j1]
                            to_do.remove(j1)
                        d[j1] = h

        except StopIteration:
            # Augment
            print("Augmenting %d" % j1)
            for k in ready:
                temp = v[k]
                v[k] = v[k] + d[k] - umin
                print("v[%d] %f -> %f" % (k, temp, v[k]))
            while True:
                iii = pred[j1]
                print("y[%d] %d -> %d" % (j1, y[j1], iii))
                y[j1] = iii
                j1, x[iii] = x[iii], j1
                if iii == i:
                    break
    #
    # Re-establish slackness since we didn't pay attention to u
    #
    for i in range(n):
        j = x[i]
        u[i] = c[i,j] - v[j]
