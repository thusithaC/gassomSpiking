#include "ToMatlab.h"
#include <stdio.h>
#include <string.h>

Engine *ep;
void tomat::start()
{
	if (!(ep = engOpen("\0"))) {
		fprintf(stderr, "\nCan't start MATLAB engine\n");
		return;
	}
	//engEvalString(ep,"cd 'D:\\ZHAOYU\\MPil\\Record_by_Date\\2013-02-19_CUDA_HappyNewYear\\matlabProgramForCudaTest';");
}

void tomat::close()
{
	engClose(ep);
}


void tomat::pushMat(Tmat &A, string name)
{
	int nrows = A.n_rows;
	int ncols = A.n_cols;
	int dim = nrows*ncols;
	double *temp = new double[dim] ;
		
	for(int i=0;i<dim;i++)
		temp[i] = A.at(i);
	  
	const char *s = name.c_str();
	push(temp,nrows,ncols,s,0);

	delete[] temp;
}
void tomat::pushCube(fcube &A, string name)
{
	int nrows = A.n_rows;
	int ncols = A.n_cols;
	int nlices = A.n_slices;
	int dim = nrows*ncols*nlices;
	double *temp = new double[dim] ;
		
	for(int i=0;i<dim;i++)
		temp[i] = A.at(i);
	  
	const char *s = name.c_str();
	push(temp,nrows,ncols,s,0);

	delete[] temp;
}
void tomat::getMat(Tmat &A, string name)
{
	const char *s = name.c_str();
	int nrows = A.n_rows;
	int ncols = A.n_cols;
	int dim = nrows*ncols;
	double *temp = new double[dim] ;

	get(temp,s,0);
	for(int i=0; i<dim;i++)
		A(i)=temp[i];
	delete[] temp; 

}


