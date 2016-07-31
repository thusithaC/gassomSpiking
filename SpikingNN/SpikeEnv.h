#pragma once


#include "configs.h"
#include "spikeData.h"
#include "tempotron.h"
#include "GASSOM_Spk.h"


class SpikeEnv
{
public:
	GASSOM_spk enc;
	Tempotron *tmpNetwork; 
	EnvParams params;
	SpikeData data;
	vector <SpikeData> data_vec;
	vector <int> data_labels;
	vector <float> proj_errors;

private:
	int nblocks_x; 
	int nblocks_y;
	int nblocks;
	int dim_basis;
	
	int count1; //counter used for various purposes. Now for MNIST data loading

	Tmat syn_inputs;
	Tmat in_spike_count;
	Tmat out_spike_count;
	Tmat tot_spike_count;
	


public:
	SpikeEnv(void);
	SpikeEnv(GASSOM_Params &gparams, EnvParams &eparams);
	SpikeEnv(EnvParams &eparams);
	~SpikeEnv(void);
	void startLearningL1(void);
	void startLearning(int i);
	Tmat getCoeffsHisto(void);
	Tmat getCoeffsHisto(int i);
	void saveDatatoMat(void);
	void readSpikeTrainData(string name);
	void getMNISTHisto(string data_file, string save_file, int ndata, int offset);
	void getMNISTSpikes(string data_file, string save_file, int ndata, int offset);
	void learnFromMNIST(string data_file, int ndata);
	void learnFromFaces(string data_file, int ndata);
	void getFacesSpikes(string data_file, string save_file);
	void getFacesHisto();
	void getLettersHisto();
	void readMNISTData(int num);
	void readMNISTDataPad(int num,int spatial_range=10);
	void readFacesData(int num);
	void getFireHistory(void);
	void processl1spikes(void);
	void processl1spikes(int i);
	float testLearning();
	void learnTempotron(string data_file, Tempotron_params &tparams, int ndata, bool pool=true, bool validate=false);
	void testTempotron(string data_file, string weights_file, Tempotron_params &tparams, int ndata, bool pool=true);
private:

};

