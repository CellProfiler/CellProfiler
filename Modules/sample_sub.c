/* sampleBasedEntropy.cpp -  */

#include <math.h>
#include "mex.h"

/* Input Arguments */
#define FITFUN_IN      prhs[0]
#define IB_IN           prhs[1]
#define REIDX_IN        prhs[2]
#define NDATA_IN        prhs[3]

/* Output Arguments */
#define DF_OUT        plhs[0]

void mexFunction( int nlhs, mxArray *plhs[], 
                  int nrhs, const mxArray*prhs[] )
     
{ 
  double *fitfun, *ib, *ndata;
  uint32_T *reidx;
  int M, num_samples, num_bases, i, j;
  double *df;

  /* Check for proper number of arguments */
  if (nrhs != 4) { 
    mexErrMsgTxt("Four input arguments required."); 
  } else if (nlhs > 1) {
    mexErrMsgTxt("Too many output arguments."); 
  } 

  num_samples = mxGetM(FITFUN_IN); 
  num_bases = mxGetN(IB_IN);

  /* Create matrices for the return arguments */ 
  DF_OUT = mxCreateDoubleMatrix(1, num_bases, mxREAL); 
  df = mxGetPr(DF_OUT);
    
  /* Assign pointers to the various parameters */ 
  fitfun = mxGetPr(FITFUN_IN);
  ib = mxGetPr(IB_IN);
  ndata = mxGetPr(NDATA_IN);
  reidx = mxGetData(REIDX_IN);

  M = 100;

#define IJ(i,j) (num_samples*(j)+(i))
#define DLOGNDATA(i,j)     (-ib[IJ(reidx[i]-1,j)] / fitfun[reidx[i]-1])

  for (j = 0; j < num_bases; j++) {
    df[j] = 0.0;
    
    for (i = 0; i < num_samples - M; i++) {
      df[j] += (DLOGNDATA(i+M,j) - DLOGNDATA(i,j)) / (ndata[i+M] - ndata[i]);
    }
  }

  return;
}
