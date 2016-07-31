//This is the main file. 
//
//
//

#include <iostream>
#include <armadillo>
#include"GASSOM_Spk.h"
#include "SpikeEnv.h"
#include "toMatlab.h"

using namespace std;
using namespace arma;



void defineParams(GASSOM_Params &gparams, EnvParams &eparams);
void trainL1basis(SpikeEnv &env);
void getMNISTSpikes(SpikeEnv &env);
void getL1Spikes(SpikeEnv &env);
void getL2Spikes(SpikeEnv &env);
void tempotronClassify(SpikeEnv &env);
void tempotronClassifyTest(SpikeEnv &env);

int main(int argc, char** argv)
{
	GASSOM_Params gparams;
	EnvParams eparams; 
	int seed = 1;

	if (argc > 1)
	{
		seed = atof(argv[1]);
		
	}
	cout<< "Seed " << seed << endl;
	arma_rng::set_seed(seed);
	
	tomat::start();
	string matlab_work_directory = "C:\\Users\\Thusitha\\Dropbox\\Public\\Spiking_GASSOM";
	string mat_set_dir = "cd('"+matlab_work_directory+"');"; 
	engEvalString(ep,mat_set_dir.c_str());
	engEvalString(ep,"clear all; clc;");

	defineParams(gparams,eparams);
	SpikeEnv env(gparams,eparams);


	//{ These are the different configurations. Uncomment the required and comment all others. Make sure the correct configuration is selected in the configs.h file. 
	
	trainL1basis(env);
	//getL1Spikes(env);
	//getMNISTSpikes(env);
	//tempotronClassify(env);
	//tempotronClassifyTest(env);

	//}

	int key=0;
	cout<<"Enter key to exit..."<<endl;
	cin>>key;

  return 0;
}



void defineParams(GASSOM_Params &gparams, EnvParams &eparams)
{


	eparams.SENSOR_WIDTH_X = SENSOR_WIDTH_X; //128:basis train, 40:MNIST, 240 faces 
	eparams.SENSOR_WIDTH_Y = SENSOR_WIDTH_Y;//128:basis train, 40:MNIST, 320 faces
	eparams.DIM_BLOCK = DIM_BLOCK;
	eparams.STRIDE = STRIDE;
	eparams.NTIMES = NTIMES;
	eparams.RESETFAC = RESETFAC;
	eparams.LAMBDA_F = LAMBDA_F;
	eparams.FIRETHRESH = FIRETHRESH; //20 for mnist, 100 for basis training , 50 for faces
	eparams.NPOLARITIES = NPOLARITIES;

	// have to calculate BASISDIM and BATCH_SIZE 
	int nblocks_x = (eparams.SENSOR_WIDTH_X -eparams.DIM_BLOCK)/eparams.STRIDE + 1;
	int nblocks_y = (eparams.SENSOR_WIDTH_Y -eparams.DIM_BLOCK)/eparams.STRIDE + 1;
	int nblocks = nblocks_x*nblocks_y;
	int dim_basis = eparams.DIM_BLOCK*eparams.DIM_BLOCK*eparams.NPOLARITIES;

	gparams.TOPO_SUBSPACE = TOPO_SUBSPACE;
	gparams.SUBNUM = SUBNUM;
	gparams.NBPERSUB = NBPERSUB;
	gparams.BASISDIM = dim_basis;
	gparams.ALPHA_A = ALPHA_A;
	gparams.ALPHA_C = ALPHA_C;
	gparams.SIGMA_A = SIGMA_A;
	gparams.SIGMA_C = SIGMA_C;
	gparams.SIGMA_N =SIGMA_N;
	gparams.SIGMA_W = SIGMA_W;
	gparams.TCONST = TCONST;
	gparams.LAMBDA_SAC = LAMBDA_SAC;
	gparams.UPDATE_FREQ = UPDATE_FREQ;
	gparams.BATCH_SIZE = nblocks;
	gparams.SLOWNESS_TRANS = SLOWNESS_TRANS;

}


void trainL1basis(SpikeEnv &env)
{
	string data_file = "data\\data_objects_4.mat";
	env.readSpikeTrainData(data_file);
	env.startLearningL1();
	env.saveDatatoMat();
#if GETSPIKEHISTORY
	env.getFireHistory();
#endif
}

void trainL2basis(SpikeEnv &env)
{
	env.readSpikeTrainData("spikes_l1_10_64_O4_s4_NG.mat");
	env.startLearningL1();
	env.saveDatatoMat();
	env.getFireHistory();
}

void getL1Spikes(SpikeEnv &env)
{
	string saved_bases_file = "data\\basis_64_10_o4_slow.mat";
	string input_events_file = "data\\data_objects_4.mat";
	env.enc.loadBasis(saved_bases_file);
	env.readSpikeTrainData(input_events_file);
	env.processl1spikes();
	env.saveDatatoMat();
#if GETSPIKEHISTORY
	env.getFireHistory();
#endif
}


void getL2Spikes(SpikeEnv &env)
{
	env.enc.loadBasis("basis_l2_256_8_p36_04_s4_tau_100_T_15_slow.mat");
	env.readSpikeTrainData("spikes_l1_10_64_O4_s4_NG.mat");
	env.processl1spikes();
	env.saveDatatoMat();
#if GETSPIKEHISTORY
	env.getFireHistory();
#endif
}


void getMNISTSpikes(SpikeEnv &env)
{
	const int num_samples = 10000;
#if 1 //L1
	string basis = "data\\basis_64_10_o4_slow.mat"; //l1 basis basis_25_16_f.mat
	string data_file = "data\\data_train.mat"; //l0 spikes
	string save_file = "data\\MNIST_S_10_64_TRAIN_ALL_S4_slow_O4basis.mat";
#elif 0//L2
	string basis = "basis_l2_256_8_p64_04_s4_tau_120_T_20_noslow2nd.mat"; //l2 basis
	string data_file = "MNIST_S_10_64_TRAIN_ALL_S4_slow_O4basis_100ms.mat"; //"MNIST_S_10_25_TEST_ALL.mat"; //l1 spikes
	string save_file = "MNIST_L2_S_256_8_TRAIN_ALL_slow_tau_120_tmp.mat"; 

#elif 0 //L3
	string basis = "basis_l3_25_4_p100_mnist.mat"; //l2 basis
	string data_file = "MNIST_L2_S_100_ALL_TRAIN.mat"; //l1 spikes
	string save_file = "MNIST_L3_S_16_TRAIN_small.mat"; 
#endif
	

	env.enc.loadBasis(basis);
    env.getMNISTSpikes(data_file, save_file, num_samples, 0); 
	//env.getFireHistory();

#if GETSPIKEHISTORY
	env.getFireHistory();
#endif
	
}


void tempotronClassifyPooled(SpikeEnv &env)
{

	string data_file = "MNIST_S_10_64_TRAIN_ALL_S4_slow_O4basis_100ms.mat"; //l1 spikes
	int ndata = 20000;

	Tempotron_params tparams;
	tparams.LRATE = 0.0001;
	tparams.MRATE = 0.5;
	tparams.TAU1 = 80000; //20->55
	tparams.TAU2 = tparams.TAU1/4;
	tparams.VT = 20;
	tparams.VR = 1;
	tparams.EVENT_TYPES = 64;
	tparams.SPATIAL_RANGE[0]=10;
	tparams.SPATIAL_RANGE[1]=10;
	tparams.N = 64;
	tparams.NUM_NEURONS = 10;
	tparams.dt = 10;
	tparams.batch_size = 100;
	tparams.iterations = 5000;


	env.learnTempotron(data_file,tparams,ndata); // this is the basic pooled version


}


void tempotronClassify(SpikeEnv &env)
{ //this is the no-pooling training code. the input spatial size is 10x10 (the output events span this spatial range)

#ifdef TEMPOTRON_
	string data_file = "data\\MNIST_S_10_64_TRAIN_ALL_S4_slow_O4basis_100ms.mat"; //l1 spikes
	int ndata = 20000; //how many instances ar eused for the training. Max 60,000
	const int spatial_range = SPATIAL_RANGE_; //hardcoded, 10x10 range for the input events
	Tempotron_params tparams;
	tparams.LRATE = LRATE_; //[10]
	tparams.MRATE = MRATE_;
	tparams.TAU1 = TAU1_; //20->55 [80,40, 160, 120]
	tparams.TAU2 = TAU2_;
	tparams.VT = VT_;
	tparams.VR = VR_;
	tparams.EVENT_TYPES = EVENT_TYPES_;
	tparams.SPATIAL_RANGE[0]=spatial_range;
	tparams.SPATIAL_RANGE[1]=spatial_range;
	tparams.N = tparams.EVENT_TYPES*spatial_range*spatial_range; //number of afferents
	tparams.NUM_NEURONS = NUM_NEURONS_;
	tparams.dt = DT_;
	tparams.batch_size = BATCH_SIZE_;
	tparams.iterations = ITERATIONS_;
	tparams.val_ratio = VAL_RATIO_;

	env.learnTempotron(data_file,tparams,ndata, false, true); // this is the basic pooled version
#else
	cout<< "Tempotron network not configured" <<endl;
	return;
#endif
}


void tempotronClassifyTest(SpikeEnv &env)
{ //this is the no-pooling training code. the input spatial size is 10x10 (the output events span this spatial range)
	#ifdef TEMPOTRON_
	string data_file = "data\\MNIST_S_10_64_TEST_ALL_S4_slow_O4basis_100ms.mat"; //l1 spikes
	string test_file = "data\\20160727_065301_Tempotron_train.mat"; //l1 spikes
	int ndata = 10000;
	const int spatial_range = SPATIAL_RANGE_; //hardcoded, 10x10 range for the input events
	Tempotron_params tparams;
	tparams.LRATE = LRATE_;
	tparams.MRATE = MRATE_;
	tparams.TAU1 = TAU1_; //20->55 [80,40, 160, 120]
	tparams.TAU2 = TAU2_;
	tparams.VT = VT_;
	tparams.VR = VR_;
	tparams.EVENT_TYPES = EVENT_TYPES_;
	tparams.SPATIAL_RANGE[0]=spatial_range;
	tparams.SPATIAL_RANGE[1]=spatial_range;
	tparams.N = tparams.EVENT_TYPES*spatial_range*spatial_range; //number of afferents
	tparams.NUM_NEURONS = NUM_NEURONS_;
	tparams.dt = DT_;
	tparams.batch_size = BATCH_SIZE_;
	tparams.iterations = ITERATIONS_;
	tparams.val_ratio = VAL_RATIO_;

	env.testTempotron(data_file,test_file,tparams,ndata, false); // this is the basic pooled version
#else
	cout<< "Tempotron network not configured" <<endl;
	return;
#endif
	
}