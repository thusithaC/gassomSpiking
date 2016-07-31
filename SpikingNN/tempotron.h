#pragma once


#include "configs.h"
#include "spikeData.h"
#include <queue>




class Tempotron
{
private:
	Tmat weights;
	Tmat grads;
	Tmat grads_prev;
	vector<SpikeData> &data_vec; // A reference to the the data vector
	vector<int> &data_labels; 
	Tempotron_params params;
	

public: 
	Tempotron(Tempotron_params pr,vector<SpikeData>& dv, vector<int> & lv);
	void train(bool pool=true, bool validate=false);
	void test(int start_idx=0,bool pool=true);
	void classify();
	void saveWeights();
	void loadWeights(Tmat &tWeights);
	~Tempotron();

};
