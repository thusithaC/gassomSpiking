#pragma once

#include "configs.h"
#include "spikeData.h"
/*
#define TOPO_SUBSPACE 5
#define SUBNUM int(TOPO_SUBSPACE*TOPO_SUBSPACE)
#define BASISDIM 512

//learning
#define ALPHA_A float(0.0050) //0.005
#define ALPHA_C float(0.0005)
#define SIGMA_A float(2.5)
#define SIGMA_C float(0.1)
#define SIGMA_N float(0.2) //0.2
#define SIGMA_W float(2) //2
#define TCONST float(1000) 


//decay times
#define lambda_sac float(1000)
*/

// structure for holding parameters
struct GASSOM_Params
{
	int TOPO_SUBSPACE;
	int SUBNUM ;
	int NBPERSUB;
	int BASISDIM;
	int BATCH_SIZE;
	float ALPHA_A;
	float ALPHA_C;
	float SIGMA_A;
	float SIGMA_C;
	float SIGMA_N;
	float SIGMA_W;
	float TCONST;
	float SLOWNESS_TRANS;

	float LAMBDA_SAC; 

	int UPDATE_FREQ;

};


class GASSOM_spk
{

//variables
private:
	GASSOM_Params params;
	float const_w;
	float const_n;


public:
	Tmat basis1;
	Tmat basis2;
	Tmat t_last_sp;
	Tmat node_prob;
	int iter;

	int acc_index;
	int update_freq;
	Tmat s_proj;
	Tmat s_x;
	Tmat s_coef1;
	Tmat s_coef2;
	Tmat s_np;

	vector < vector<float> > dts;

//methods
private:
	void orthonormalise(Tmat &A,Tmat &B); //take respective columns from A and B, and orthonormalize
	void normalizeProb(Tmat &A);
	void preprocessInput(Tmat &A);
	
public:
	GASSOM_spk(void);
	GASSOM_spk(GASSOM_Params params);
	~GASSOM_spk();
	int GASSOMEncode(Tmat &X, int s_index, float t_now, bool onlyEncode=true);
	void GASSOMUpdate();
	Tmat getConvolMatrix(float sigma);
	Tmat genTransProbSpk(float dt, float gamma);
	GASSOM_Params getparams(void) {return params;}	
	void loadBasis(string name);
	void saveBasis(void);
	void resetEncoder(void);
	float projError(Tmat &X);


};