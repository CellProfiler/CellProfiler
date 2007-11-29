/*
 * Matlab function which calls rwalkpthread.
 *
 */

#include <math.h>
#include <vlcutils/pgm.h>
#include "mex.h"
#include "rwalk.h"

#define IMAGE_IN prhs[0]
#define RESTARTPROB_IN prhs[1]
#define NWALKS_IN prhs[2]
#define SEED_IN prhs[3]

#define PMASK_OUT plhs[0]

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
     unsigned int m, n;
     struct pgm16 *image, *pmask;
     int nwalks, i, j, x, y, nworkers;
     double *seed_values, *pmask_values, restart_prob, value, threshold;
     void *image_data;
     struct seed *seed;

     if (nrhs != 4)
          mexErrMsgTxt("Too few input arguments.");
     if (nlhs > 1)
          mexErrMsgTxt("Too many output arguments.");

     m = mxGetM(IMAGE_IN);
     n = mxGetN(IMAGE_IN);
     /* Reconstruct image from input. */
     if (!mxIsNumeric(IMAGE_IN))
          mexErrMsgTxt("Image must be numeric.");
     if (mxGetNumberOfDimensions(IMAGE_IN) != 2)
          mexErrMsgTxt("Image must be two-dimensional.");
     image = make_pgm16(n, m);
     image_data = mxGetData(IMAGE_IN);
     switch (mxGetClassID(IMAGE_IN)) {
     case mxDOUBLE_CLASS:
          for (i = 0; i < m; i++)
               for (j = 0; j < n; j++) {
                    value = ((double *)image_data)[j * m + i];
                    if (value < 0 || value > 1)
                         mexErrMsgTxt("Image of type double must be within "
                                      "the range [0, 1].");
                    pgm_pixel(image, j, i) = 255 * value + 0.5;
               }
          break;
     case mxUINT8_CLASS:
          for (i = 0; i < m; i++)
               for (j = 0; j < n; j++)
                    pgm_pixel(image, j, i) = ((unsigned char *)image_data)[j * m + i];
          break;
     default:
          mexErrMsgTxt("Image must be of class uint8 or double.");
     }
     /* Check that restart probability is sane. */
     if (!mxIsNumeric(RESTARTPROB_IN))
          mexErrMsgTxt("Restart probability must be numeric.");
     if (mxGetM(RESTARTPROB_IN) != 1 || mxGetN(RESTARTPROB_IN) != 1)
          mexErrMsgTxt("Restart probability must be scalar.");
     restart_prob = mxGetScalar(RESTARTPROB_IN);
     if (restart_prob < 0 || restart_prob > 1)
          mexErrMsgTxt("Restart probability must be in the range [0, 1].");
     /* Check that the number of walks is sane. */
     if (!mxIsNumeric(NWALKS_IN))
          mexErrMsgTxt("Number of walks must be numeric.");
     if (mxGetM(NWALKS_IN) != 1 || mxGetN(NWALKS_IN) != 1)
          mexErrMsgTxt("Number of walks must be scalar.");
     nwalks = (int)mxGetScalar(NWALKS_IN);
     if (nwalks < 1)
          mexErrMsgTxt("Number of walks must be positive.");
     /* Reconstruct seeds from input. */
     if (!mxIsNumeric(SEED_IN))
          mexErrMsgTxt("Seed must be numeric.");
     if (mxGetN(SEED_IN) != 2)
          mexErrMsgTxt("Seed must have two columns (x and y).");
     seed = make_seed(mxGetM(SEED_IN));
     seed_values = mxGetPr(SEED_IN);
     for (i = 0; i < seed->size; i++) {
          x = seed_values[0 * seed->size + i];
          y = seed_values[1 * seed->size + i];
          if (y < 0 || y > m || x < 0 || x > n)
               mexErrMsgTxt("Seed must consist of points within the "
                            "image boundaries.");
          set_seed(seed, i, point(x, y));
     }
     
     nworkers = 2;
     pmask = rwalkpthread(image, seed, (int)mxGetScalar(NWALKS_IN), 0, 
                          restart_prob, nworkers);

#if 0
     /* Threshold at the minimum stationary probability within the seeds. */
     for (i = 0; i < seed->size; i++) {
          value = pgm_pixel(pmask, seed->points[i].x, seed->points[i].y);
          if (i == 0 || value < threshold)
               threshold = value;
     }
#endif
#if 1
     /* Global normalization. */
     threshold = 0;
     for (i = 0; i < m; i++)
          for (j = 0; j < n; j++)
               if (pgm_pixel(pmask, j, i) > threshold)
                    threshold = pgm_pixel(pmask, j, i);
#endif

     /* Prepare output matrix. */
     PMASK_OUT = mxCreateDoubleMatrix(m, n, mxREAL);
     pmask_values = mxGetPr(PMASK_OUT);
     for (i = 0; i < m; i++)
          for (j = 0; j < n; j++) {
               value = pgm_pixel(pmask, j, i) / threshold;
               pmask_values[j * m + i] = value <= 1 ? value : 1;
          }
     free(image);
     free(pmask);
     return;
}
