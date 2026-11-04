# setuptools: language = c++

from numpy cimport *
from libcpp.vector cimport vector
import numpy as np

from cpython cimport *

import_array()

cdef extern from "flow_utils.hpp":
    cdef enum FLOW_TYPE_T:
        NO_FLOW,
        WITHOUT_TRANSHIPMENT_FLOW,
        WITHOUT_EXTRA_MASS_FLOW
        
cdef extern from "fastemd_hat.hpp" nogil:
    cdef:
        NUM_T emd_hat_gd_metric_no_flow[NUM_T](
            vector[NUM_T] &P,
            vector[NUM_T] &Q,
            vector[vector[NUM_T]] &C,
            NUM_T extra_mass_penalty,
            vector[vector[NUM_T]] *F)
        NUM_T emd_hat_gd_metric_without_transshipment_flow[NUM_T](
            vector[NUM_T] &P,
            vector[NUM_T] &Q,
            vector[vector[NUM_T]] &C,
            NUM_T extra_mass_penalty,
            vector[vector[NUM_T]] *F)
        NUM_T emd_hat_gd_metric_without_extra_mass_flow[NUM_T](
            vector[NUM_T] &P,
            vector[NUM_T] &Q,
            vector[vector[NUM_T]] &C,
            NUM_T extra_mass_penalty,
            vector[vector[NUM_T]] *F)
        NUM_T emd_hat_no_flow[NUM_T](
            vector[NUM_T] &P,
            vector[NUM_T] &Q,
            vector[vector[NUM_T]] &C,
            NUM_T extra_mass_penalty,
            vector[vector[NUM_T]] *F)
        NUM_T emd_hat_without_transshipment_flow[NUM_T](
            vector[NUM_T] &P,
            vector[NUM_T] &Q,
            vector[vector[NUM_T]] &C,
            NUM_T extra_mass_penalty,
            vector[vector[NUM_T]] *F)
        NUM_T emd_hat_without_extra_mass_flow[NUM_T](
            vector[NUM_T] &P,
            vector[NUM_T] &Q,
            vector[vector[NUM_T]] &C,
            NUM_T extra_mass_penalty,
            vector[vector[NUM_T]] *F)

cdef extern from "npy_helpers.hpp" nogil:
    cdef void np1D_to_vector[T](ndarray obj, vector[T] &v)
    cdef void np2D_to_vector[T](ndarray obj, vector[vector[T]] &v)
    cdef void vector_to_np2D[T](ndarray obj, vector[vector[T]] &v)

'''Don't calculate flow'''
EMD_NO_FLOW = NO_FLOW
'''Calculate flow but ignore all bins with edges larger than max(C)'''
EMD_WITHOUT_TRANSHIPMENT_FLOW = WITHOUT_TRANSHIPMENT_FLOW
'''Calculate all flow except for the extra mass bin'''
EMD_WITHOUT_EXTRA_MASS_FLOW = WITHOUT_EXTRA_MASS_FLOW
   
def emd_hat_int32(p, q, c, 
                  extra_mass_penalty=None,
                  flow_type=EMD_NO_FLOW, gd_metric=False):
    '''Calculate earth-mover's distance on 32-bit integers
    :param p: first histogram - a 1d numpy array of source "amounts of dirt"
    
    :param q: second histogram - a 1d numpy array of destination "amounts of dirt"
    
    :param c: a 2d array of shape (len(p), len(q)) giving the distance from each
              bin of p to each bin of q
              
    :param extra_mass_penalty: a penalty multiplier assessed if there's more
        dirt in p than in q or vice-versa. Default is the maximum distance that
        the dirt could move.
        
    :param flow_type: one of the following
    
        EMD_NO_FLOW don't calculate flow
        
        EMD_WITHOUT_TRANSHIPMENT_FLOW fills f with flows between
            all bins connected with edges smaller than max(C)
            
        PEMD_WITHOUT_EXTRA_MASS_FLOW fills f with flows between all
            bins (except the imaginary extra mass bin).
            
    :param gd_metric: True if the P/Q distances are a metric. If so, please
            define to speed things up. 
            See http://en.wikipedia.org/wiki/Metric_(mathematics)
    
    :returns: The minimum cost of moving the dirt from p to q and, if
              the flow type is EMD_NO_FLOW, a len(p) by len(q) matrix
              of the dirt's comings and goings.
    '''
    cdef:
        vector[int32_t] vp
        vector[int32_t] vq
        vector[vector[int32_t]] vc
        vector[vector[int32_t]] vf
        vector[vector[int32_t]] *pvf = NULL
        int32_t emp = -1
        int32_t result
        Py_ssize_t plen = len(p)
        Py_ssize_t qlen = len(q)
        Py_ssize_t idx
    
    assert plen == c.shape[0], "cost matrix first dimension must be the same as first histogram size"
    assert qlen == c.shape[1], "cost matrix second dimension must be the same as second histogram size"
    np1D_to_vector[int32_t](np.ascontiguousarray(p, np.int32), vp)
    np1D_to_vector[int32_t](np.ascontiguousarray(q, np.int32), vq)
    np2D_to_vector[int32_t](np.ascontiguousarray(c, np.int32), vc)
    #
    # The FastEMD code only works on square matrices. I am not sure,
    # but I think zero "dirt" only costs space, so we resize the smaller
    # of the two vectors and the cost matrix (the value for rows or
    # columns of zero dirt doesn't matter).
    #
    if plen > qlen:
        vq.resize(plen)
        for 0<=idx<plen:
            vc[idx].resize(plen)
    elif qlen > plen:
        vp.resize(qlen)
        vc.resize(qlen, vector[int32_t](qlen))
        
    if extra_mass_penalty is not None:
        emp = int(extra_mass_penalty)
    if flow_type == EMD_NO_FLOW:
        if gd_metric:
            result = emd_hat_gd_metric_no_flow[int32_t](
                vp, vq, vc, emp, pvf)
        else:
            result = emd_hat_no_flow[int32_t](vp, vq, vc, emp, pvf)
        return result
    vf.resize(vp.size(), vector[int32_t](vp.size()))
    pvf = &vf
    if flow_type == EMD_WITHOUT_TRANSHIPMENT_FLOW:
        if gd_metric:
            result = emd_hat_gd_metric_without_transshipment_flow[int32_t](
                vp, vq, vc, emp, pvf)
        else:
            result = emd_hat_without_transshipment_flow[int32_t](vp, vq, vc, emp, pvf)
    else:
        if gd_metric:
            result = emd_hat_gd_metric_without_extra_mass_flow[int32_t](
                vp, vq, vc, emp, pvf)
        else:
            result = emd_hat_without_extra_mass_flow[int32_t](vp, vq, vc, emp, pvf)
    f = np.zeros((len(p), len(q)), np.int32)
    if plen < vf.size():
        vf.resize(plen)
    if qlen < vf[0].size():
        for 0<=idx<plen:
            vf[idx].resize(qlen)
    vector_to_np2D[int32_t](f, vf)
    return result, f
