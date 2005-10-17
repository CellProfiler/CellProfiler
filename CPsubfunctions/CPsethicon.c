#include <windows.h> 
#include <string.h> 
#include <shellapi.h> 
#include "mex.h" 
 
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray 
*prhs[]) 
{ 
  if ((nrhs == 2)) 
    { 
        char *iconpath; 
int buflength; 
HICON hicon; 

/* get Icon handle */ 
buflength = mxGetN(prhs[1])+1; 
iconpath = mxCalloc(buflength,sizeof(char)); 
mxGetString(prhs[1], iconpath, buflength); 
 
if (hicon = (HICON)ExtractIcon(NULL, iconpath, 0)) 
{ 
 
//Get window handle 
HWND hwnd; 
char *windowname; 
 
buflength = mxGetN(prhs[0])+1; 
windowname = mxCalloc(buflength,sizeof(char)); 
mxGetString(prhs[0], windowname, buflength); 
 
//Beep(600,2000); 

if (hwnd = FindWindow(NULL,windowname)) { 
//Beep(1000,2000); 
SendMessage(
      (HWND) hwnd,
      (UINT) WM_SETICON,
      (WPARAM) ICON_SMALL,
      (LPARAM) hicon
);
SendMessage(
      (HWND) hwnd,
      (UINT) WM_SETICON,
      (WPARAM) ICON_BIG,
      (LPARAM) hicon
);
} 
} 

    } 
}