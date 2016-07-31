#include "GASSOM_Spk.h"

#include "toMatlab.h"

GASSOM_spk::GASSOM_spk()
{

}
GASSOM_spk::~GASSOM_spk()
{

}


GASSOM_spk::GASSOM_spk(GASSOM_Params params):params(params)
{


	basis1 = Tmat(params.BASISDIM, params.SUBNUM, fill::randn);
	basis2 = Tmat(params.BASISDIM, params.SUBNUM, fill::randn);
	orthonormalise(basis1,basis2);

	update_freq = params.UPDATE_FREQ;
	
	t_last_sp = Tmat(params.BATCH_SIZE,1,fill::zeros);
	node_prob = Tmat(params.SUBNUM,params.BATCH_SIZE,fill::randu);
	normalizeProb(node_prob);
	acc_index = 0;
	iter = 0;

	s_proj = Tmat(params.SUBNUM, update_freq, fill::zeros);
	s_x = Tmat(params.BASISDIM, update_freq, fill::zeros);
	s_coef1 = Tmat(params.SUBNUM, update_freq, fill::zeros);
	s_coef2 = Tmat(params.SUBNUM, update_freq, fill::zeros);
	s_np = Tmat(params.SUBNUM, update_freq, fill::zeros);

	dts = vector< vector<float> >(params.BATCH_SIZE);

	const_w = 1/(2*params.SIGMA_W*params.SIGMA_W);
	const_n = 1/(2*params.SIGMA_N*params.SIGMA_N);
}

void GASSOM_spk::loadBasis(string bname)
{
	string bn = "load('"+ bname +"');";
	engEvalString(ep,bn.c_str());
	
	double *d_basis1,*d_basis2;
	d_basis1 = new double[params.BASISDIM*params.SUBNUM];
	d_basis2 = new double[params.BASISDIM*params.SUBNUM];
	tomat::get(d_basis1,"basis1",0);
	tomat::get(d_basis2,"basis2",0);

	Tmat b1(d_basis1,params.BASISDIM,params.SUBNUM,true, true);
	Tmat b2(d_basis2,params.BASISDIM,params.SUBNUM,true, true);

	basis1 = b1;
	basis2 = b2;

	delete[] d_basis1;
	delete[] d_basis2;
}

void GASSOM_spk::saveBasis()
{
	tomat::pushMat(basis1,"basis1");
	tomat::pushMat(basis2,"basis2");
}

void GASSOM_spk::orthonormalise(Tmat &A, Tmat &B)
{
	A = normalise(A);
	B = B - A % repmat(sum(A%B),params.BASISDIM,1);
	B = normalise(B);

}


void GASSOM_spk::normalizeProb(Tmat &A)
{
	Tmat sum_probs = repmat(sum(A),params.SUBNUM,1);
	A = A/sum_probs;
}

void GASSOM_spk::preprocessInput(Tmat &A)
{
	A = A - repmat(mean( A ),params.BASISDIM,1);
	A = normalise(A);
}


Tmat GASSOM_spk::genTransProbSpk(float dt, float lambda)
{
	float tdecay = exp(-lambda*dt);
	float pUni = (1-params.SLOWNESS_TRANS*tdecay)/params.SUBNUM; 
	Tmat pTable = eye(params.SUBNUM,params.SUBNUM)*(params.SLOWNESS_TRANS*tdecay); 
	pTable = pTable+pUni;
	return pTable;
}



Tmat GASSOM_spk::getConvolMatrix(float sigma)
{
	Tmat T = zeros<Tmat>(params.SUBNUM,params.SUBNUM);
	
	for(int j=0; j<params.SUBNUM; j++)
	{
		for(int i=0; i<params.SUBNUM; i++)
		{
			float dr = (j%params.TOPO_SUBSPACE -i%params.TOPO_SUBSPACE)*(j%params.TOPO_SUBSPACE -i%params.TOPO_SUBSPACE);
			float dc = (j/params.TOPO_SUBSPACE -i/params.TOPO_SUBSPACE)*(j/params.TOPO_SUBSPACE -i/params.TOPO_SUBSPACE);

			T(j,i) = exp(-(dr+dc)/(2*sigma*sigma));

		}
	}
	return T;
}

int GASSOM_spk::GASSOMEncode(Tmat &X, int s_index, float t_now,bool onlyEncode)
{
	preprocessInput(X);

	float dt = t_now - t_last_sp(s_index);
	Tmat tp = genTransProbSpk(dt, params.LAMBDA_SAC);
	t_last_sp(s_index) = t_now;

	Tmat coef1 = basis1.t()*X;
	Tmat coef2 = basis2.t()*X;
	Tmat proj  = coef1%coef1 + coef2%coef2;
	Tmat perr = ones<Tmat>(params.SUBNUM,1)- proj;
	//checked

#if GETSPIKEHISTORY
	dts[s_index].push_back(dt);				 
#endif

	//tomat::pushMat(t_last_sp,"t_last");
	//tomat::pushMat(tp,"tp");

	Tmat emiss_prob=exp(-proj*const_w) % exp(-perr*const_n);
	Tmat np = (tp.t()*node_prob.col(s_index)) % emiss_prob;
	normalizeProb(np);
	node_prob.col(s_index) = np;	
	uword  winner;

	np.max(winner); //winner for the updates
	Tmat win_prob = zeros<Tmat>(params.SUBNUM,1);
	win_prob(winner) = 1;
	
	if(!onlyEncode) //savestuff for updates
	{
		s_proj.col(acc_index) = proj;
		s_x.col(acc_index) = X;
		s_coef1.col(acc_index) = coef1;
		s_coef2.col(acc_index) = coef2;
		s_np.col(acc_index) = win_prob;
		acc_index++;
	}

	//recalculate a winner for getting the representation
 uword  winner_sig;
#if GREEDYWINNER
	proj.max(winner_sig);
#else
	np.max(winner_sig);
#endif

	return (int)winner_sig;
}


void GASSOM_spk::GASSOMUpdate()
{
	if(acc_index==params.UPDATE_FREQ)
	{
		
		float sigma = (params.SIGMA_A*exp(-float(iter)/params.TCONST)+params.SIGMA_C);
		float alpha = (params.ALPHA_A*exp(-float(iter)/params.TCONST)+params.ALPHA_C);
	
		//Smoothing,
		Tmat G = getConvolMatrix(sigma);
		s_np = G*s_np;

		Tmat n_const = 1/(sqrt(s_proj)+EPS);
		n_const = n_const % s_np;

		Tmat w_c1 = n_const % s_coef1;
		Tmat w_c2 = n_const % s_coef2;
		Tmat winput1 = s_x*w_c1.t();
		Tmat winput2 = s_x*w_c2.t();

		Tmat diff1 = winput1 - basis1%repmat(sum(w_c1%s_coef1,1).t(),params.BASISDIM,1)
							 - basis2%repmat(sum(w_c1%s_coef2,1).t(),params.BASISDIM,1);

		Tmat diff2 = winput2 - basis1%repmat(sum(w_c2%s_coef1,1).t(),params.BASISDIM,1)
							 - basis2%repmat(sum(w_c2%s_coef2,1).t(),params.BASISDIM,1);
	
	
		basis1 = basis1 + alpha*diff1;
		basis2 = basis2 + alpha*diff2;

		orthonormalise(basis1,basis2);

		iter++;
		acc_index = 0;
	}
		//tomat::pushMat(basis1,"b1");
	//tomat::pushMat(basis2,"b2");
}


/*return the minimum prjection error on a subspace*/
float GASSOM_spk::projError(Tmat &X)
{
	preprocessInput(X);

	Tmat coef1 = basis1.t()*X;
	Tmat coef2 = basis2.t()*X;
	Tmat proj  = coef1%coef1 + coef2%coef2;
	float max_proj;

	max_proj = proj.max();
	return 1-max_proj;
}

void GASSOM_spk::resetEncoder()
{
	t_last_sp = Tmat(params.BATCH_SIZE,1,fill::zeros);
	node_prob = Tmat(params.SUBNUM,params.BATCH_SIZE,fill::randu);
	normalizeProb(node_prob);
	acc_index = 0;
	
}