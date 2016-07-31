#pragma once

#include <armadillo>
#include <stddef.h>


using namespace std;
using namespace arma;
 
#define GETSPIKEHISTORY 1 // this will dump some statistics about the spiking history during the training. Useful in setting parameters. 
#define LOADBASIS 0 // not used
#define GREEDYWINNER 1 // if set to '1' , this will calculate the winner based only on projection error for encoding purposes. 


#define EPS float(0.000000001)

typedef arma::mat Tmat;
typedef arma::Mat<int> Imat;
typedef unsigned char uchar;
typedef unsigned int uint;
typedef unsigned short  ushort; 

struct EnvParams
{
	int STRIDE;
	int DIM_BLOCK;
	int SENSOR_WIDTH_X;
	int SENSOR_WIDTH_Y;
	float RESETFAC;
	int NTIMES;
	float LAMBDA_F;
	float FIRETHRESH;
	int NPOLARITIES;
	
};



struct Tempotron_params
{
	float LRATE;
	float MRATE;
	float TAU1;
	float TAU2; //Tau2=Tau1/4;
	float VT; //threshold
	float VR; //rest
	int EVENT_TYPES;
	int SPATIAL_RANGE[2];
	int N; // afferents
	int NUM_NEURONS; 
	float dt;
	int batch_size;
	float val_ratio; //0-1 value for the percentage of validation data to be used.
	int iterations;
	

	//init in the Tempotron constructor
	float delay_to_max; 
	float V0; //initilized later 
};

/*Alway remember to change this when running different files
A little bit akward, but cant help
*/

//typedef ushort udata; //for faces data! 
//typedef uchar udata; // for MNIST and L! basis training data


/***********************PARAMS**********************************/
	

#if 1/* For L1 basis training*/                                                                                                                           
	typedef uchar udata;  //   { A little bit akward. But cant help. If If the spatial dimention of the inpu tis <256 pixels use this. Or use 'typedef ushort udata;' This is Useful in controlling the data size.
	#define DATA_IS_UCHAR 1 // {  If the spatial dimention of the inpu tis <256 pixels use 1. Else use 0. }
	#define CHECKERROR 0 //For checking the projexction error in training
	const int SENSOR_WIDTH_X = 128; //Spatial dimension of the sensor 
	const int SENSOR_WIDTH_Y = 128;//Spatial dimension of the sensor 
	const int DIM_BLOCK = 10; // size of the local patch being processed
	const int STRIDE = 4; // stride between local patches
	const int NTIMES = 5; // how many times to cycle the same data. If data size is small, use a bigger number here. 
	const float RESETFAC = 0.001; // Close to Zero. 
	const float LAMBDA_F = 1.0/10000; // Time constant of the leaky integrators. You can adjust it based on the event density. 
	const int FIRETHRESH = 40; // The threshold for generating an output event. 

	const int TOPO_SUBSPACE = 8; // This controls the number of subspaces. We assume a square lettice for the GASSOM with TOPO_SUBSPACE*TOPO_SUBSPACE units in total
	const int SUBNUM = TOPO_SUBSPACE*TOPO_SUBSPACE;
	const int NBPERSUB = 2; //fixed. 
	const float ALPHA_A = 0.1; // Exponentailly decaying learning rate
	const float ALPHA_C = 0.001; // Constant learning rate 
	const float SIGMA_A = 2; // Exponentailly decaying Gaussian neighborhhod update factor
	const float SIGMA_C = 0.0001; // constant Gaussian neighborhhod update factor. Setting this to a small value would be helpful in specilization
	const float SIGMA_N = 0.25; //{ These two doesnt have to be changed IF the norm of the inputs are 1. but if the inputs to the GASSOM are not 1 norm, and these params 
	const float SIGMA_W = 2; //   { are not changed, things would go wrong. So KEEP INPUT 1 NORMED, or try tuning these params. Rule is SIGMA_N << SIGMA_W
	const float TCONST = 10000; //Timecosntant for the exponential decay in learning /neighborhood rates. This is in number of events. 
	const float LAMBDA_SAC = 1.0/100e3; // Time constant for slowness 
	const int UPDATE_FREQ = 10; // The basis vectors will be updated after this number of output events. 
	const float SLOWNESS_TRANS = 1; //1 or 0. Settign this to zero will  train a model without slowness. 
	const int NPOLARITIES = 2; // Number of inpu tevent types. This can be changed when trining higher layers with inclreased input types. 

#elif 0/* L2 basis training*/
	
	typedef uchar udata; 
	#define DATA_IS_UCHAR 1
	#define CHECKERROR 0
	const int SENSOR_WIDTH_X = 32; //128:basis train, 40:MNIST, 240 faces 
	const int SENSOR_WIDTH_Y = 32;//128:basis train, 40:MNIST, 320 faces
	const int DIM_BLOCK = 8;
	const int STRIDE = 2; //2
	const int NTIMES = 1;
	const float RESETFAC = 0.001;
	const float LAMBDA_F = 1.0/100000; //60000
	const int FIRETHRESH = 10; //40 was good for O4,8x8
	const int NPOLARITIES = 64;

	const int TOPO_SUBSPACE = 16;
	const int SUBNUM = TOPO_SUBSPACE*TOPO_SUBSPACE;
	const int NBPERSUB = 2;
	const float ALPHA_A = 0.2;
	const float ALPHA_C = 0.01;
	const float SIGMA_A = 4;
	const float SIGMA_C = 0.1;
	const float SIGMA_N = 0.25;
	const float SIGMA_W = 2;
	const float TCONST = 40000;
	const float LAMBDA_SAC = 1.0/10e3;
	const int UPDATE_FREQ = 10;
	const float SLOWNESS_TRANS = 1;

#elif 0/* MNIST get L1 spikes*/
	#define DATA_IS_UCHAR 1
	typedef uchar udata; 
	const int SENSOR_WIDTH_X = 40; //40:MNIST16 
	const int SENSOR_WIDTH_Y = 40;//40:MNIST16, 
	const int DIM_BLOCK = 10;
	const int STRIDE = 4;
	const int NTIMES = 1;
	const float RESETFAC = 0.001;
	const float LAMBDA_F = 1.0/10000;
	const int FIRETHRESH = 40; //20 for mnist, 100 for basis training , 50 for faces
	const int NPOLARITIES = 2;

	const int TOPO_SUBSPACE = 8;
	const int SUBNUM = TOPO_SUBSPACE*TOPO_SUBSPACE;
	const int NBPERSUB = 2;
	const float ALPHA_A = 0.2;
	const float ALPHA_C = 0.02;
	const float SIGMA_A = 2;
	const float SIGMA_C = 0.0001;
	const float SIGMA_N = 0.25;
	const float SIGMA_W = 2;
	const float TCONST = 2000;
	const float LAMBDA_SAC = 1.0/10e4;
	const int UPDATE_FREQ = 10;
	const float SLOWNESS_TRANS = 0;


#elif 0 /*for MNIST, l2 spikes */
	#define DATA_IS_UCHAR 1
	typedef uchar udata; 
	const int SENSOR_WIDTH_X = 24; //128:basis train, 40:MNIST, 240 faces 
	const int SENSOR_WIDTH_Y = 24;//128:basis train, 40:MNIST, 320 faces
	const int DIM_BLOCK = 8;
	const int STRIDE = 2;
	const int NTIMES = 1;
	const float RESETFAC = 0.001;
	const float LAMBDA_F = 1.0/100000;
	const int FIRETHRESH = 10; //20 for mnist, 100 for basis training , 50 for faces
	const int NPOLARITIES = 64;

	const int TOPO_SUBSPACE = 16;
	const int SUBNUM = TOPO_SUBSPACE*TOPO_SUBSPACE;
	const int NBPERSUB = 2;
	const float ALPHA_A = 0.2;
	const float ALPHA_C = 0.02;
	const float SIGMA_A = 4;
	const float SIGMA_C = 0.0001;
	const float SIGMA_N = 0.25;
	const float SIGMA_W = 2;
	const float TCONST = 40000;
	const float LAMBDA_SAC = 1.0/80000;
	const int UPDATE_FREQ = 10;
	const float SLOWNESS_TRANS = 1;


#elif 0 /* For MNIST tempotron network training with 1st layer outputs from the GASSOM */
#define TEMPOTRON_ 1
	typedef uchar udata; 
	#define DATA_IS_UCHAR 1
	//fields related to Tempotron
	const int SPATIAL_RANGE_ = 9; // the spatial range of the input events to the tempotron netwrk. This depends on the settings used in generating the GASSOM based repreresentation
	const float LRATE_ = 0.0001*10; //learning rate
	const float MRATE_ = 0.5; //The momentum value for the Gradiant descent 
	const float TAU1_ = 80000; //in us
	const float TAU2_ = TAU1_/4; // fixed based on the original publication
	const float VT_ = 20; // Spikeing threshold
	const float VR_ = 1; //resting potential
	const int EVENT_TYPES_ = 64; // output event types from the GASSOM used to encode the input
	const int NUM_NEURONS_ = 10 ; // number of classes 
	const float DT_ =10;
	const int BATCH_SIZE_ = 100; //mini batch size used for training
	const int VAL_RATIO_ = 0.2; //how much of the data is used for validation
	const int ITERATIONS_ = 100; //how many time to recycle the dat ain training the network


	//other fields, no need to change. Have to include to prevent the compiler from complaining, but not used in tempotron training. <Bad programming :( Didnt have time to clean this up. Th eTempotron network was later added>
	#define CHECKERROR 0
	const int SENSOR_WIDTH_X = 32; //128:basis train, 40:MNIST, 240 faces 
	const int SENSOR_WIDTH_Y = 32;//128:basis train, 40:MNIST, 320 faces
	const int DIM_BLOCK = 8;
	const int STRIDE = 2; //2
	const int NTIMES = 1;
	const float RESETFAC = 0.001;
	const float LAMBDA_F = 1.0/100000; //60000
	const int FIRETHRESH = 10; //40 was good for O4,8x8
	const int NPOLARITIES = 64;

	const int TOPO_SUBSPACE = 16;
	const int SUBNUM = TOPO_SUBSPACE*TOPO_SUBSPACE;
	const int NBPERSUB = 2;
	const float ALPHA_A = 0.2;
	const float ALPHA_C = 0.01;
	const float SIGMA_A = 4;
	const float SIGMA_C = 0.1;
	const float SIGMA_N = 0.25;
	const float SIGMA_W = 2;
	const float TCONST = 40000;
	const float LAMBDA_SAC = 1.0/10e3;
	const int UPDATE_FREQ = 10;
	const float SLOWNESS_TRANS = 1;




#endif