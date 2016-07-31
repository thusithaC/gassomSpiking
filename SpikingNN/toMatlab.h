
#pragma once
#include <engine.h>
#include <iostream>
#include "configs.h"


extern Engine *ep;
namespace tomat
{
	void start();
	void close();
	void pushMat(Tmat &A, string name);
	void getMat(Tmat &A, string name);
	void pushCube(fcube &A, string name);

	//void push(const double* m,int rowNum,int colNum, char* c);
	template<class T>
	void push(const T* m,int rowNum,int colNum,const char* c,bool trans = 0)
	{
		double* store = new double[rowNum*colNum];
		std::copy(m,m+rowNum*colNum,store);	
	
		mxArray *mT;
		if(trans)
		{ 
			mT = mxCreateDoubleMatrix(colNum, rowNum, mxREAL);
			memcpy((void *)mxGetPr(mT), store, mxGetNumberOfElements(mT)*sizeof(double));
			engPutVariable(ep, c, mT);
			
			char str[100];
			sprintf(str,"%s = %s'",c,c);
			engEvalString(ep,str);
		}else
		{
			mT = mxCreateDoubleMatrix(rowNum,colNum, mxREAL);
			memcpy((void *)mxGetPr(mT), store, mxGetNumberOfElements(mT)*sizeof(double));
			engPutVariable(ep, c, mT);
		}
		mxDestroyArray(mT);
		delete[] store;

	}



	template<class T>
	int get(T* m,const char* c,bool trans = 0)
	{
		char str[100];
		mxArray *mT;
		if(trans)
		{
			sprintf(str,"tmpForCppFatch = %s'",c);
			engEvalString(ep,str);
			mT = engGetVariable(ep, "tmpForCppFatch");
		}
		else
			mT = engGetVariable(ep, c);

		if(mT==NULL)
		{
			std::cout<<"\n*************\n"<<
				"Error Loading values from Matlab"<<"\n***************\n"<<endl; 
			return -1;; 
		}
		int size = mxGetNumberOfElements(mT);
		T* store = new T[size];
		memcpy(store,(void *)mxGetPr(mT), size*sizeof(T));
		std::copy(store,store+size,m);

		delete[] store;
		mxDestroyArray(mT);
		return 0;
	}


}
