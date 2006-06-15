/* propagate.c - */

/* CellProfiler is distributed under the GNU General Public License.
 * See the accompanying file LICENSE for details.                   
 *                                                                  
 * Developed by the Whitehead Institute for Biomedical Research.    
 * Copyright 2003,2004,2005.                                        
 *                                                                  
 * Authors:                                                         
 *   Anne Carpenter <carpenter@wi.mit.edu>                          
 *   Thouis Jones   <thouis@csail.mit.edu>                          
 *   In Han Kang    <inthek@mit.edu>                                
 *
 * $Revision$
 */

#include <math.h>
#include "mex.h"
#include <queue>
#include <vector>
#include <iostream>
using namespace std;

/* Input Arguments */
#define LABELS_IN       prhs[0]
#define IM_IN           prhs[1]
#define MASK_IN         prhs[2]
#define LAMBDA_IN       prhs[3]

/* Output Arguments */
#define LABELS_OUT        plhs[0]
#define DISTANCES_OUT        plhs[1]

#define IJ(i,j) ((j)*m+(i))

class Pixel { 
public:
  double distance;
  unsigned int i, j;
  double label;
  Pixel (double ds, unsigned int ini, unsigned int inj, double l) : 
    distance(ds), i(ini), j(inj), label(l) {}
};

struct Pixel_compare { 
 bool operator() (const Pixel& a, const Pixel& b) const 
 { return a.distance > b.distance; }
};

typedef priority_queue<Pixel, vector<Pixel>, Pixel_compare> PixelQueue;

static double
clamped_fetch(double *image, 
              int i, int j,
              int m, int n)
{
  if (i < 0) i = 0;
  if (i >= m) i = m-1;
  if (j < 0) j = 0;
  if (j >= n) j = n-1;

  return (image[IJ(i,j)]);
}

static double
Difference(double *image,
           int i1,  int j1,
           int i2,  int j2,
           unsigned int m, unsigned int n,
           double lambda)
{
  int delta_i, delta_j;
  double pixel_diff = 0.0;

  /* At some point, the width over which differences are calculated should probably be user controlled. */
  for (delta_j = -1; delta_j <= 1; delta_j++) {
    for (delta_i = -1; delta_i <= 1; delta_i++) {
      pixel_diff += fabs(clamped_fetch(image, i1 + delta_i, j1 + delta_j, m, n) - 
                         clamped_fetch(image, i2 + delta_i, j2 + delta_j, m, n));
    }
  }

  return (sqrt(pixel_diff*pixel_diff + (fabs((double) i1 - i2) + fabs((double) j1 - j2)) * lambda * lambda));
}

static void
push_neighbors_on_queue(PixelQueue &pq, double dist,
                        double *image,
                        unsigned int i, unsigned int j,
                        unsigned int m, unsigned int n,
                        double lambda, double label)
{
  /* 4-connected */
  if (i > 0) {
    pq.push(Pixel(dist + Difference(image, i, j, i-1, j, m, n, lambda), i-1, j, label));
  }                                                                   
  if (j > 0) {                                                        
    pq.push(Pixel(dist + Difference(image, i, j, i, j-1, m, n, lambda), i, j-1, label));
  }                                                                   
  if (i < (m-1)) {                                                    
    pq.push(Pixel(dist + Difference(image, i, j, i+1, j, m, n, lambda), i+1, j, label));
  }                                                                   
  if (j < (n-1)) {                                                    
    pq.push(Pixel(dist + Difference(image, i, j, i, j+1, m, n, lambda), i, j+1, label));
  } 

  /* 8-connected */
  if ((i > 0) && (j > 0)) {
    pq.push(Pixel(dist + Difference(image, i, j, i-1, j-1, m, n, lambda), i-1, j-1, label));
  }                                                                       
  if ((i < (m-1)) && (j > 0)) {                                           
    pq.push(Pixel(dist + Difference(image, i, j, i+1, j-1, m, n, lambda), i+1, j-1, label));
  }                                                                       
  if ((i > 0) && (j < (n-1))) {                                           
    pq.push(Pixel(dist + Difference(image, i, j, i-1, j+1, m, n, lambda), i-1, j+1, label));
  }                                                                       
  if ((i < (m-1)) && (j < (n-1))) {                                       
    pq.push(Pixel(dist + Difference(image, i, j, i+1, j+1, m, n, lambda), i+1, j+1, label));
  }
  
}

static void propagate(double *labels_in, double *im_in,
                      mxLogical *mask_in, double *labels_out,
                      double *dists,
                      unsigned int m, unsigned int n,
                      double lambda)
{
  unsigned int i, j;
  PixelQueue pixel_queue;

  for (j = 0; j < n; j++) {
    for (i = 0; i < m; i++) {
      dists[IJ(i,j)] = mxGetInf();
      
      double label = labels_in[IJ(i,j)];
      labels_out[IJ(i,j)] = label;
      if ((label > 0) && (mask_in[IJ(i,j)])) {
        dists[IJ(i,j)] = 0.0;
        push_neighbors_on_queue(pixel_queue, 0.0, im_in, i, j, m, n, lambda, label);
      }
    }
  }

  while (! pixel_queue.empty()) {
    Pixel p = pixel_queue.top();
    pixel_queue.pop();
    
    //    cout << "popped " << p.i << " " << p.j << endl;

    
    if (! mask_in[IJ(p.i, p.j)]) continue;
    //    cout << "going on\n";

    if ((dists[IJ(p.i, p.j)] > p.distance) && (mask_in[IJ(p.i,p.j)])) {
      dists[IJ(p.i, p.j)] = p.distance;
      labels_out[IJ(p.i, p.j)] = p.label;
      push_neighbors_on_queue(pixel_queue, p.distance, im_in, p.i, p.j, m, n, lambda, p.label);
    }
  }
}

void mexFunction( int nlhs, mxArray *plhs[], 
                  int nrhs, const mxArray*prhs[] )
     
{ 
    double *labels_in, *im_in; 
    mxLogical *mask_in;
    double *labels_out, *dists;
    double *lambda; 
    mxArray *dists_array;
    unsigned int m, n; 
    
    /* Check for proper number of arguments */
    
    if (nrhs != 4) { 
        mexErrMsgTxt("Four input arguments required."); 
    } else if (nlhs > 2) {
        mexErrMsgTxt("Too many output arguments."); 
    } 

    m = mxGetM(IM_IN); 
    n = mxGetN(IM_IN);

    if ((m != mxGetM(LABELS_IN)) ||
        (n != mxGetN(LABELS_IN))) {
      mexErrMsgTxt("First and second arguments must have same size.");
    }

    if ((m != mxGetM(MASK_IN)) ||
        (n != mxGetN(MASK_IN))) {
      mexErrMsgTxt("First and third arguments must have same size.");
    }

    if (! mxIsDouble(LABELS_IN)) {
      mexErrMsgTxt("First argument must be a double array.");
    }
    if (! mxIsDouble(IM_IN)) {
      mexErrMsgTxt("Second argument must be a double array.");
    }
    if (! mxIsLogical(MASK_IN)) {
      mexErrMsgTxt("Third argument must be a logical array.");
    }

    /* Create matrices for the return arguments */ 
    LABELS_OUT = mxCreateDoubleMatrix(m, n, mxREAL); 
    dists_array = mxCreateDoubleMatrix(m, n, mxREAL);
    
    /* Assign pointers to the various parameters */ 
    labels_in = mxGetPr(LABELS_IN);
    im_in = mxGetPr(IM_IN);
    mask_in = mxGetLogicals(MASK_IN);
    labels_out = mxGetPr(LABELS_OUT);
    lambda = mxGetPr(LAMBDA_IN);

    /* Do the actual computations in a subroutine */
    dists = mxGetPr(dists_array);

    propagate(labels_in, im_in, mask_in, labels_out, dists, m, n, *lambda); 

    if (nlhs == 2) {
      DISTANCES_OUT = dists_array;
    } else {
      mxDestroyArray(dists_array);
    }      

    return;
}
