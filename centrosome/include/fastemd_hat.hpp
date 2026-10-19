/* fastemd_hat.hpp - helper templates to bridge between FastEMD and Cython
 *
 * CellProfiler is distributed under the GNU General Public License,
 * but this file is licensed under the more permissive BSD license.
 * See the accompanying file LICENSE for details.
 *
 * Copyright (c) 2003-2009 Massachusetts Institute of Technology
 * Copyright (c) 2009-2015 Broad Institute
 * All rights reserved.
 *
 * Please see the AUTHORS file for credits.
 *
 * Website: http://www.cellprofiler.org
 *
 * This file provides templates of all the given flow types
 * so that Cython only has templates of the numeric types.
 */
#include "emd_hat.hpp"

template<typename NUM_T>
NUM_T emd_hat_gd_metric_no_flow(
    const std::vector<NUM_T> & P,
    const std::vector<NUM_T> & Q,
    const std::vector< std::vector<NUM_T> > &C,
    NUM_T extra_mass_penalty=-1,
    std::vector< std::vector<NUM_T> > *F=NULL)
{
    return emd_hat_gd_metric<NUM_T>()(P, Q, C, extra_mass_penalty, F);
}

template<typename NUM_T>
NUM_T emd_hat_gd_metric_without_transshipment_flow (
    const std::vector<NUM_T> & P,
    const std::vector<NUM_T> & Q,
    const std::vector< std::vector<NUM_T> > &C,
    NUM_T extra_mass_penalty=-1,
    std::vector< std::vector<NUM_T> > *F=NULL) 
{
    return emd_hat_gd_metric<NUM_T, WITHOUT_TRANSHIPMENT_FLOW>()(
             P, Q, C, extra_mass_penalty, F);
}

template<typename NUM_T>
NUM_T emd_hat_gd_metric_without_extra_mass_flow(
    const std::vector<NUM_T> & P,
    const std::vector<NUM_T> & Q,
    const std::vector< std::vector<NUM_T> > &C,
    NUM_T extra_mass_penalty=-1,
    std::vector< std::vector<NUM_T> > *F=NULL) 
{
    return emd_hat_gd_metric<NUM_T, WITHOUT_EXTRA_MASS_FLOW>()(
             P, Q, C, extra_mass_penalty, F);
}

template<typename NUM_T>
NUM_T emd_hat_no_flow(
    const std::vector<NUM_T> & P,
    const std::vector<NUM_T> & Q,
    const std::vector< std::vector<NUM_T> > &C,
    NUM_T extra_mass_penalty=-1,
    std::vector< std::vector<NUM_T> > *F=NULL) 
{
    return emd_hat<NUM_T>()(P, Q, C, extra_mass_penalty, F);
}

template<typename NUM_T>
NUM_T emd_hat_without_transshipment_flow(
    const std::vector<NUM_T> & P,
    const std::vector<NUM_T> & Q,
    const std::vector< std::vector<NUM_T> > &C,
    NUM_T extra_mass_penalty=-1,
    std::vector< std::vector<NUM_T> > *F=NULL) 
{
    return emd_hat<NUM_T, WITHOUT_TRANSHIPMENT_FLOW>()(
             P, Q, C, extra_mass_penalty, F);
}

template<typename NUM_T>
NUM_T emd_hat_without_extra_mass_flow(
    const std::vector<NUM_T> & P,
    const std::vector<NUM_T> & Q,
    const std::vector< std::vector<NUM_T> > &C,
    NUM_T extra_mass_penalty=-1,
    std::vector< std::vector<NUM_T> > *F=NULL) 
{
    return emd_hat<NUM_T, WITHOUT_EXTRA_MASS_FLOW>()(
             P, Q, C, extra_mass_penalty, F);
}

