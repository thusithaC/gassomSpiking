#include "SpikeEnv.h"
#include "toMatlab.h"



SpikeEnv::SpikeEnv(void)
{
	tmpNetwork = NULL;
}


SpikeEnv::~SpikeEnv(void)
{
	if(tmpNetwork != NULL)
		delete tmpNetwork;
}

SpikeEnv::SpikeEnv(GASSOM_Params &gparams, EnvParams &eparams):
	enc(gparams),params(eparams)
{

	nblocks_x = (eparams.SENSOR_WIDTH_X -eparams.DIM_BLOCK)/eparams.STRIDE + 1;
	nblocks_y = (eparams.SENSOR_WIDTH_Y -eparams.DIM_BLOCK)/eparams.STRIDE + 1;
	nblocks = nblocks_x*nblocks_y;
	dim_basis = gparams.BASISDIM;

	syn_inputs = zeros<Tmat>(dim_basis,nblocks);
	in_spike_count = zeros<Tmat>(nblocks,1);
	out_spike_count = zeros<Tmat>(nblocks,1);
	tot_spike_count = zeros<Tmat>(nblocks,1);
	count1 = 1;
	//check some stuff;
	assert(dim_basis==gparams.BASISDIM);
	assert(nblocks==gparams.BATCH_SIZE);
	
	tmpNetwork = NULL;
}


SpikeEnv::SpikeEnv(EnvParams &eparams):params(eparams){}


/*Basis learning in L1 from our dataset */
void SpikeEnv::startLearningL1(void)
{
	int r_loc, c_loc, spk_event, l_max_r,l_max_c, l_min_r,l_min_c, nfire=0;
	uint t_ls,t_now;
	vector <int> l_c, l_r;

	int ncapture = 20;
	int interval = (data.nevents*params.NTIMES)/ncapture ;

	for (int iter=0; iter<params.NTIMES; iter++)
	{
		t_ls = 0;
		enc.resetEncoder();
		for(int e=0; e<data.nevents;e++)
		{
			l_c.clear(); 
			l_r.clear(); 

			// R -> Y, C-> X
			c_loc = data.x[e]; 
			r_loc = data.y[e];
			spk_event = data.p[e];
			t_now = data.ts[e];

			if (r_loc>=SENSOR_WIDTH_Y || c_loc > SENSOR_WIDTH_X)
				cout<< "Error in setting sensor width"<<endl;

			l_max_r = r_loc/params.STRIDE;
			l_max_r = (l_max_r>=nblocks_y)?nblocks_y-1:l_max_r;
			l_max_c = c_loc/params.STRIDE;
			l_max_c = (l_max_c>=nblocks_x)?nblocks_x-1:l_max_c;

			l_min_r = ceil(max(r_loc-params.DIM_BLOCK+1,0)/(float)params.STRIDE);
			l_min_c = ceil(max(c_loc-params.DIM_BLOCK+1,0)/(float)params.STRIDE);

			for(int i=l_min_r; i<l_max_r+1;i++)
				l_r.push_back(i);
		
			for(int i=l_min_c; i<l_max_c+1;i++)
				l_c.push_back(i);

			//should calculate for all combinations of l_c and l_r
			for (vector<int>::iterator it_r = l_r.begin() ; it_r != l_r.end(); ++it_r)
			{
				for (vector<int>::iterator it_c = l_c.begin() ; it_c != l_c.end(); ++it_c)
				{
					int r_idx, c_idx, r_blk_idx, c_blk_idx , blk_lin_idx, local_lin_idx;
					int pswitch;

					r_blk_idx = *it_r;
					c_blk_idx = *it_c;
					r_idx = r_loc - r_blk_idx*params.STRIDE;
					c_idx = c_loc - c_blk_idx*params.STRIDE;
					blk_lin_idx = c_blk_idx*nblocks_y + r_blk_idx;
					local_lin_idx = c_idx*params.DIM_BLOCK + r_idx;
				
					pswitch = spk_event*params.DIM_BLOCK*params.DIM_BLOCK;

					syn_inputs.col(blk_lin_idx) *= exp(-params.LAMBDA_F*(t_now-t_ls));   
					syn_inputs(local_lin_idx+pswitch,blk_lin_idx)+=1;	

#if GETSPIKEHISTORY
					in_spike_count(blk_lin_idx,0)+=1;
#endif
					if (sum(syn_inputs.col(blk_lin_idx)) > params.FIRETHRESH)
					{
						//cout<< "FIRE!\n ";
						
						Tmat X =  syn_inputs.col(blk_lin_idx);					
						syn_inputs.col(blk_lin_idx) *= params.RESETFAC;

						//tomat::pushMat(X,"X");

						//call the encoder
						enc.GASSOMEncode(X,blk_lin_idx,t_now,0);
						enc.GASSOMUpdate();
						nfire++;
#if GETSPIKEHISTORY
						out_spike_count(blk_lin_idx,0)+=1;
						
#endif
					}
				

				}
			}//end of all instances;
			t_ls = t_now;
		
			if(((e+iter*data.nevents)%interval)==0)
			{
				tomat::pushMat(enc.basis1,"basis1");
				tomat::pushMat(enc.basis2,"basis2");
				tomat::pushMat(out_spike_count,"spikes_out");
				tomat::pushMat(in_spike_count,"spikes_in");
				cout << float(e+iter*data.nevents)/(data.nevents*params.NTIMES)*100<<" Total Fire:"<<nfire<<endl;


#if CHECKERROR
				proj_errors.push_back(testLearning());
				tomat::push(proj_errors.data(),proj_errors.size(),1,"proj_errors",0);
#endif
			}
		}//end of event
	}//end of iter

	//todo: push the projerror to mattlab
	tomat::push(proj_errors.data(),proj_errors.size(),1,"proj_errors",0);
}

void SpikeEnv::startLearning(int index)
{
	int r_loc, c_loc, spk_event, l_max_r,l_max_c, l_min_r,l_min_c;
	uint t_ls,t_now;
	vector <int> l_c, l_r;
	SpikeData sdata = data_vec[index];

		t_ls = 0;
		enc.resetEncoder();
		for(int e=0; e<sdata.nevents;e++)
		{
			l_c.clear(); 
			l_r.clear(); 

			// R -> Y, C-> X
			c_loc = sdata.x[e]-1; //convert matlab 1 based indexing to 0 based 
			r_loc = sdata.y[e]-1;
			spk_event = sdata.p[e]-1; //event polarity is given as 2,1 in the data file 
			t_now = sdata.ts[e];

			if (r_loc>=SENSOR_WIDTH_Y || c_loc > SENSOR_WIDTH_X)
				cout<< "Error in setting sensor width"<<endl;
			
			l_max_r = r_loc/params.STRIDE;
			l_max_r = (l_max_r>=nblocks_y)?nblocks_y-1:l_max_r;
			l_max_c = c_loc/params.STRIDE;
			l_max_c = (l_max_c>=nblocks_x)?nblocks_x-1:l_max_c;

			l_min_r = ceil(max(r_loc-params.DIM_BLOCK+1,0)/(float)params.STRIDE);
			l_min_c = ceil(max(c_loc-params.DIM_BLOCK+1,0)/(float)params.STRIDE);

			for(int i=l_min_r; i<l_max_r+1;i++)
				l_r.push_back(i);
		
			for(int i=l_min_c; i<l_max_c+1;i++)
				l_c.push_back(i);

			

			//should calculate for all combinations of l_c and l_r
			for (vector<int>::iterator it_r = l_r.begin() ; it_r != l_r.end(); ++it_r)
			{
				for (vector<int>::iterator it_c = l_c.begin() ; it_c != l_c.end(); ++it_c)
				{
					int r_idx, c_idx, r_blk_idx, c_blk_idx , blk_lin_idx, local_lin_idx;
					int pswitch;

					r_blk_idx = *it_r;
					c_blk_idx = *it_c;
					r_idx = r_loc - r_blk_idx*params.STRIDE;
					c_idx = c_loc - c_blk_idx*params.STRIDE;
					blk_lin_idx = c_blk_idx*nblocks_y + r_blk_idx;
					local_lin_idx = c_idx*params.DIM_BLOCK + r_idx;
				
					pswitch = spk_event*params.DIM_BLOCK*params.DIM_BLOCK;

					syn_inputs.col(blk_lin_idx) *= exp(-params.LAMBDA_F*(t_now-t_ls));   
					syn_inputs(local_lin_idx+pswitch,blk_lin_idx)+=1;	

					

#if GETSPIKEHISTORY
					in_spike_count(blk_lin_idx,0)+=1;
#endif
					if (sum(syn_inputs.col(blk_lin_idx)) > params.FIRETHRESH)
					{
						//cout<< "FIRE!\n ";
	
						Tmat X =  syn_inputs.col(blk_lin_idx);					
						syn_inputs.col(blk_lin_idx) *= params.RESETFAC;

						//tomat::pushMat(X,"X");

						//call the encoder
						enc.GASSOMEncode(X,blk_lin_idx,t_now,0);
						enc.GASSOMUpdate();
#if GETSPIKEHISTORY
						out_spike_count(blk_lin_idx,0)+=1;
#endif
					}
				

				}
			}//end of all instances;


			t_ls = t_now;
		
		}//end of event


	//todo: push the projerror to mattlab
	//tomat::push(proj_errors.data(),proj_errors.size(),1,"proj_errors",0);
}



/*Used for reading L1 and L2 training data*/
void SpikeEnv::readSpikeTrainData(string name)
{

	string load_command = "load('"+name+ "');";
	const char* data_name = load_command.c_str();
	engEvalString(ep,data_name);

	udata *xdata, *ydata, *polarity;
	uint *tsdata;
	int length=0; 

	//first get the length
	tomat::get(&length,"len",0);

	//these guys will be cleaned when the SpikeData id deleted
	xdata = new udata[length];
	ydata = new udata[length];
	polarity = new udata[length];
	tsdata = new uint[length];

	
	tomat::get(xdata,"x",0);
	tomat::get(ydata,"y",0);
	tomat::get(polarity,"p",0);
	tomat::get(tsdata,"ts",0);

	data.initilize(length,xdata,ydata,polarity,tsdata);
	engEvalString(ep,"clear all;"); //remove the data from matlab

}

void SpikeEnv::readMNISTData(int num)
{
	
	string ns = "[ x,y,p,ts,len ] = getData( data{ " + to_string(num+1) + "});";
	engEvalString(ep,ns.c_str());

	//uint8_T *xdata, *ydata, *polarity;
	//uint32_T *tsdata;
	int length=0; 
	SpikeData sdata;

	//first get the length
	tomat::get(&length,"len",0);

	//these guys will be cleaned when the SpikeData id deleted
	sdata.x = new udata[length];
	sdata.y = new udata[length];
	sdata.p = new udata[length];
	sdata.ts = new uint[length];
	sdata.nevents = length;
	
	tomat::get(sdata.x,"x",0);
	tomat::get(sdata.y,"y",0);
	tomat::get(sdata.p,"p",0);
	tomat::get(sdata.ts,"ts",0);

	//data.clear();
	data_vec.push_back(sdata);

	//data.initilize(length,xdata,ydata,polarity,tsdata);
	
	//sdata.clear();
	//engEvalString(ep,"clear all;"); //remove the data from matlab

}

void SpikeEnv::readMNISTDataPad(int num, int spatial_range)
{//put a dummy node 100ms after the final event. 
	
	int PADX=spatial_range, PADY=spatial_range; 
	string ns = "[ x,y,p,ts,len ] = getData( data{ " + to_string(num+1) + "});";
	engEvalString(ep,ns.c_str());

	//uint8_T *xdata, *ydata, *polarity;
	//uint32_T *tsdata;
	int length=0; 
	SpikeData sdata;

	//first get the length
	tomat::get(&length,"len",0);

	//these guys will be cleaned when the SpikeData id deleted
	sdata.x = new udata[length+1];
	sdata.y = new udata[length+1];
	sdata.p = new udata[length+1];
	sdata.ts = new uint[length+1];
	sdata.nevents = length+1;
	
	tomat::get(sdata.x,"x",0);
	tomat::get(sdata.y,"y",0);
	tomat::get(sdata.p,"p",0);
	tomat::get(sdata.ts,"ts",0);

	//pad a datapoint at the end. This is just a workaround. Very bad coding practice
	for(int xi=0; xi<PADX; xi++)
		for(int yi=0; yi<PADY; yi++)
		{
			sdata.x[length] = xi+1;
			sdata.y[length] = yi+1;
			sdata.p[length] = 1;
			sdata.ts[length] = sdata.ts[length-1]+200000;
		}


	//data.clear();
	data_vec.push_back(sdata);

	//data.initilize(length,xdata,ydata,polarity,tsdata);
	
	//sdata.clear();
	//engEvalString(ep,"clear all;"); //remove the data from matlab

}

void SpikeEnv::getFireHistory(void)
{
	
	in_spike_count /=(dim_basis/params.NPOLARITIES);
	tomat::pushMat(in_spike_count, "spikes_in");
	tomat::pushMat(out_spike_count, "spikes_out");

	//convert vector <vector <float> > to Tmat
	int max_spikes = 0;
	for(int i=0;i<enc.dts.size(); i++)
	{
		if(enc.dts[i].size()>max_spikes)
			max_spikes = enc.dts[i].size(); 
	}

	Tmat dthistory(max_spikes,enc.dts.size(), fill::zeros); 
	
	for(int i=0;i<enc.dts.size(); i++)
	{
		for(int j=0; j<enc.dts[i].size(); j++)
		dthistory(j,i) = enc.dts[i][j];
	}

	tomat::pushMat(dthistory, "dthistory");

}

void SpikeEnv::getMNISTHisto(string data_file, string save_file, int ndata, int offset)
{

	Tmat fire_data(enc.getparams().SUBNUM,ndata, fill::zeros); 

	string df =  "load('D:\\Data\\spiking\\MNIST\\" + data_file + "');";
	string sf =  "save('" + save_file + "', 'fire_data');";
	
	engEvalString(ep,df.c_str());
		
	
	for(int im=offset; im<offset+ndata; im++ )
	{
		readMNISTData(im);
		cout << "reading data: "<< ((float)im)/ndata*100<<endl;
	}

	engEvalString(ep,"clear data;");

	int comp_count = 0;
	#pragma omp parallel for
	for(int im=offset; im<offset+ndata; im++ )
	{
		fire_data.col(im-offset) = getCoeffsHisto(im);
		comp_count++;
		cout << "processing data: "<< ((float)comp_count)/ndata*100<<endl;
	}

	tomat::pushMat(fire_data, "fire_data");
	engEvalString(ep,sf.c_str());
	

}

void SpikeEnv::learnFromMNIST(string data_file, int ndata)
{
	int comp_count = 0 , ncapture = 10;
	int interval = (ndata)/ncapture ;
	Tmat fire_data(enc.getparams().SUBNUM,ndata, fill::zeros); 
	string df =  "load('" + data_file + "');";
	
	//load the data
	engEvalString(ep,df.c_str());
	for(int im=0; im<ndata; im++ )
	{
		readMNISTData(im);
		cout << "reading data: "<< ((float)im)/ndata*100<<endl;
	}
	engEvalString(ep,"clear data;");

		
	for(int im=0; im<ndata; im++ )
	{
		startLearning(im);
		comp_count++;
		cout << "processing data: "<< ((float)comp_count)/ndata*100<<endl;
		
		if((im%interval)==0)
		{
			tomat::pushMat(enc.basis1,"basis1");
			tomat::pushMat(enc.basis2,"basis2");
		}

	}

}

void SpikeEnv::getMNISTSpikes(string data_file, string save_file, int ndata, int offset)
{
	 
	Tmat fire_data(enc.getparams().SUBNUM,ndata, fill::zeros); 
	string df =  "load('" + data_file + "');";
	string sf =  "save('" + save_file + "', 'data');";
	
	//load the data
	engEvalString(ep,df.c_str());
		
	cout <<df<<endl;
	for(int im=offset; im<offset+ndata; im++ )
	{
		readMNISTData(im);
		cout << "reading data: "<< ((float)im)/ndata*100<<endl;
	}
	engEvalString(ep,"clear data;");

	int comp_count = 0;
	
	for(int im=offset; im<offset+ndata; im++ )
	{
		processl1spikes(im);
		comp_count++;
		cout << "processing data: "<< ((float)comp_count)/ndata*100<<endl;
	}

	engEvalString(ep,sf.c_str());
	

}


void SpikeEnv::getFacesSpikes(string data_file, string save_file)
{
	 
	int ndata = 24*7;
	string df =  "load('" + data_file + "');";
	string sf =  "save('" + save_file + "', 'data');";
	
	//load the data
	engEvalString(ep,df.c_str());
	
	for(int im=0; im<ndata; im++ )
	{
		readFacesData(im);
		cout << "reading data: "<< ((float)im)/ndata*100<<endl;
	}
	engEvalString(ep,"clear data;");

	int comp_count = 0;
	
	for(int im=0; im<ndata; im++ )
	{
		processl1spikes(im);
		comp_count++;
		cout << "processing data: "<< ((float)comp_count)/ndata*100<<endl;
	}

	engEvalString(ep,sf.c_str());
	

}

void SpikeEnv::learnFromFaces(string data_file, int ndata)
{
	int comp_count = 0 , ncapture = 10;
	int interval = (ndata)/ncapture ;
	Tmat fire_data(enc.getparams().SUBNUM,ndata, fill::zeros); 
	string df =  "load('" + data_file + "');";
	
	//load the data
	engEvalString(ep,df.c_str());
	for(int im=0; im<ndata; im++ )
	{
		readFacesData(im);
		cout << "reading data: "<< ((float)im)/ndata*100<<endl;
	}
	engEvalString(ep,"clear data;");

		
	for(int im=0; im<ndata; im++ )
	{
		startLearning(im);
		comp_count++;
		cout << "processing data: "<< ((float)comp_count)/ndata*100<<endl;
		
		if((im%interval)==0)
		{
			tomat::pushMat(enc.basis1,"basis1");
			tomat::pushMat(enc.basis2,"basis2");
		}

	}

}

void SpikeEnv::getLettersHisto(void)
{
	int ndata = 72;
	int offset = 0;
	Tmat fire_data(enc.getparams().SUBNUM,ndata, fill::zeros); 
	engEvalString(ep,"load('D:\\Data\\spiking\\letters\\letters_static.mat');");
		
	

	for(int im=offset; im<offset+ndata; im++ )
	{
		readMNISTData(im);
		cout << "reading data: "<< ((float)im)/ndata*100<<endl;
	}
	engEvalString(ep,"clear data;");

	int comp_count = 0;
	#pragma omp parallel for
	for(int im=offset; im<offset+ndata; im++ )
	{
		fire_data.col(im-offset) = getCoeffsHisto(im);
		comp_count++;
		cout << "processing data: "<< ((float)comp_count)/ndata*100<<endl;
	}

	tomat::pushMat(fire_data, "fire_data");
	engEvalString(ep,"save('letters_static_36_10', 'fire_data');");
	

}


Tmat SpikeEnv::getCoeffsHisto(void)
{
	int r_loc, c_loc, spk_event, l_max_r,l_max_c, l_min_r,l_min_c;
	uint t_ls,t_now;
	vector <int> l_c, l_r;
	Tmat winner_histo(enc.getparams().SUBNUM,1,fill::zeros);

	//in_spike_count.zeros();
	//out_spike_count.zeros();
	t_ls = 0;

	for(int e=0; e<data.nevents;e++)
	{
		l_c.clear(); 
		l_r.clear(); 

		r_loc = data.y[e]-1; //convert matlab 1 based indexing to 0 based 
		c_loc = data.x[e]-1;
		spk_event = data.p[e]-1; //event polarity is given as 2,1 in the data file 
		t_now = data.ts[e];

		l_max_r = r_loc/params.STRIDE;
		l_max_r = (l_max_r>=nblocks_y)?nblocks_y-1:l_max_r;
		l_max_c = c_loc/params.STRIDE;
		l_max_c = (l_max_c>=nblocks_x)?nblocks_x-1:l_max_c;

		l_min_r = ceil(max(r_loc-params.DIM_BLOCK+1,0)/(float)params.STRIDE);
		l_min_c = ceil(max(c_loc-params.DIM_BLOCK+1,0)/(float)params.STRIDE);

		for(int i=l_min_r; i<l_max_r+1;i++)
			l_r.push_back(i);
		
		for(int i=l_min_c; i<l_max_c+1;i++)
			l_c.push_back(i);

		//should calculate for all combinations of l_c and l_r
		for (vector<int>::iterator it_r = l_r.begin() ; it_r != l_r.end(); ++it_r)
		{
			for (vector<int>::iterator it_c = l_c.begin() ; it_c != l_c.end(); ++it_c)
			{
				int r_idx, c_idx, r_blk_idx, c_blk_idx , blk_lin_idx, local_lin_idx;
				int pswitch,winner;

				r_blk_idx = *it_r;
				c_blk_idx = *it_c;
				r_idx = r_loc - r_blk_idx*params.STRIDE;
				c_idx = c_loc - c_blk_idx*params.STRIDE;
				blk_lin_idx = c_blk_idx*nblocks_x + r_blk_idx;
				local_lin_idx = c_idx*params.DIM_BLOCK + r_idx;
				
				//pswitch = (spk_event>0)?0:params.DIM_BLOCK*params.DIM_BLOCK; //tnc
				pswitch = spk_event*params.DIM_BLOCK*params.DIM_BLOCK;
				
				syn_inputs.col(blk_lin_idx) *= exp(-params.LAMBDA_F*(t_now-t_ls));   
				syn_inputs(local_lin_idx+pswitch,blk_lin_idx)+=1;	

				

				
#if GETSPIKEHISTORY
				in_spike_count(blk_lin_idx,0)+=1;
#endif
				if (sum(syn_inputs.col(blk_lin_idx)) > params.FIRETHRESH)
				{
					//cout<< "FIRE!\n ";
	
					Tmat X =  syn_inputs.col(blk_lin_idx);	
					
					syn_inputs.col(blk_lin_idx) *= params.RESETFAC;
					//tomat::pushMat(X, "X");
					//call the encoder
					winner = enc.GASSOMEncode(X,blk_lin_idx,t_now,1);
					winner_histo(winner,0)+=1;

					
#if GETSPIKEHISTORY
					out_spike_count(blk_lin_idx,0)+=1;
#endif
				}			
			
			}
		}//end of all instances;

		t_ls = t_now;
		

	}//end of event
#if GETSPIKEHISTORY
	//in_spike_count /=(dim_basis/2);
	cout << "spike ratio"<<round((out_spike_count)/((in_spike_count/(dim_basis/2))+EPS))<<endl;
#endif

	
	winner_histo = winner_histo/repmat((sum(winner_histo)+EPS),enc.getparams().SUBNUM,1);
	//tomat::pushMat(winner_histo, "winner_histo");
	//cout << winner_histo << endl;
	return winner_histo;
}


/*Return the coefficient thistogram for the spikedata in data_vec[i]*/
Tmat SpikeEnv::getCoeffsHisto(int index)
{
	int r_loc, c_loc, spk_event, l_max_r,l_max_c, l_min_r,l_min_c;
	uint t_ls,t_now;
	vector <int> l_c, l_r;
	Tmat winner_histo(enc.getparams().SUBNUM,1,fill::zeros);
	SpikeData sdata = data_vec[index];
	Tmat syn_inputs_local = zeros<Tmat>(dim_basis,nblocks);;
	in_spike_count.zeros();
	out_spike_count.zeros();
	t_ls = 0;
	enc.resetEncoder();

	for(int e=0; e<sdata.nevents;e++)
	{
		l_c.clear(); 
		l_r.clear(); 

		r_loc = sdata.y[e]-1; //convert matlab 1 based indexing to 0 based 
		c_loc = sdata.x[e]-1;
		spk_event = sdata.p[e]-1; //event polarity is given as 2,1 in the data file 
		t_now = sdata.ts[e];


		l_max_r = r_loc/params.STRIDE;
		l_max_r = (l_max_r>=nblocks_y)?nblocks_y-1:l_max_r;
		l_max_c = c_loc/params.STRIDE;
		l_max_c = (l_max_c>=nblocks_x)?nblocks_x-1:l_max_c;

		l_min_r = ceil(max(r_loc-params.DIM_BLOCK+1,0)/(float)params.STRIDE);
		l_min_c = ceil(max(c_loc-params.DIM_BLOCK+1,0)/(float)params.STRIDE);

		for(int i=l_min_r; i<l_max_r+1;i++)
			l_r.push_back(i);
		
		for(int i=l_min_c; i<l_max_c+1;i++)
			l_c.push_back(i);

		//should calculate for all combinations of l_c and l_r
		for (vector<int>::iterator it_r = l_r.begin() ; it_r != l_r.end(); ++it_r)
		{
			for (vector<int>::iterator it_c = l_c.begin() ; it_c != l_c.end(); ++it_c)
			{
				int r_idx, c_idx, r_blk_idx, c_blk_idx , blk_lin_idx, local_lin_idx;
				int pswitch,winner;

				r_blk_idx = *it_r;
				c_blk_idx = *it_c;
				r_idx = r_loc - r_blk_idx*params.STRIDE;
				c_idx = c_loc - c_blk_idx*params.STRIDE;
				blk_lin_idx = c_blk_idx*nblocks_x + r_blk_idx;
				local_lin_idx = c_idx*params.DIM_BLOCK + r_idx;
				
				//pswitch = (spk_event>0)?0:params.DIM_BLOCK*params.DIM_BLOCK; //tnc
				  pswitch = spk_event*params.DIM_BLOCK*params.DIM_BLOCK; //tnc
		
				
				syn_inputs_local.col(blk_lin_idx) *= exp(-params.LAMBDA_F*(t_now-t_ls));   
				syn_inputs_local(local_lin_idx+pswitch,blk_lin_idx)+=1;	
			

				

				
#if GETSPIKEHISTORY
				in_spike_count(blk_lin_idx,0)+=1;
#endif
				if (sum(syn_inputs_local.col(blk_lin_idx)) > params.FIRETHRESH)
				{
					//cout<< "FIRE!\n ";
	
					Tmat X =  syn_inputs_local.col(blk_lin_idx);	
					
					syn_inputs_local.col(blk_lin_idx) *= params.RESETFAC;
					
					//tomat::pushMat(X, "X");
					
					//call the encoder
					winner = enc.GASSOMEncode(X,blk_lin_idx,t_now,1);
					winner_histo(winner,0)+=1;

					
#if GETSPIKEHISTORY
					out_spike_count(blk_lin_idx,0)+=1;
#endif
				}			
			
			}
		}//end of all instances;

		t_ls = t_now;
		

	}//end of event
#if GETSPIKEHISTORY
	//in_spike_count /=(dim_basis/2);
	cout << "spike ratio"<<round((out_spike_count)/((in_spike_count/(dim_basis/2))+EPS))<<endl;
	cout << "Total out spikes " << sum(out_spike_count)<<endl;
	cout << "max spike ratio " << max(round((out_spike_count)/((in_spike_count/(dim_basis/2))+EPS)))<<endl;
#endif

	
	winner_histo = winner_histo/repmat((sum(winner_histo)+EPS),enc.getparams().SUBNUM,1);
	//tomat::pushMat(winner_histo, "winner_histo");
	//cout << winner_histo << endl;
	return winner_histo;
}




void SpikeEnv::getFacesHisto()
{
	
	int ndata = 168;
	int offset = 0;
	Tmat fire_data(enc.getparams().SUBNUM,ndata, fill::zeros); 
	engEvalString(ep,"load('D:\\Data\\spiking\\faces\\faces_s.mat');");
		
	
	for(int im=offset; im<offset+ndata; im++ )
	{
		readFacesData(im);
		cout << "reading data: "<< ((float)im)/ndata*100<<endl;
	}
	engEvalString(ep,"clear data;");

	int comp_count = 0;
	#pragma omp parallel for
	for(int im=offset; im<offset+ndata; im++ )
	{
		fire_data.col(im-offset) = getCoeffsHisto(im);
		comp_count++;
		cout << "processing data: "<< ((float)comp_count)/ndata*100<<endl;
	}

	tomat::pushMat(fire_data, "fire_data");


}


void SpikeEnv::readFacesData(int num)
{
	
	string ns = "[ x,y,p,ts,len ] = getData16( data{ " + to_string(num+1) + "});";
	engEvalString(ep,ns.c_str());

	//uint8_T *xdata, *ydata, *polarity;
	//uint32_T *tsdata;
	int length=0; 
	SpikeData sdata;

	//first get the length
	tomat::get(&length,"len",0);

	//these guys will be cleaned when the SpikeData id deleted
	sdata.x = new udata[length];
	sdata.y = new udata[length];
	sdata.p = new udata[length];
	sdata.ts = new uint[length];
	sdata.nevents = length;
	
	tomat::get(sdata.x,"x",0);
	tomat::get(sdata.y,"y",0);
	tomat::get(sdata.p,"p",0);
	tomat::get(sdata.ts,"ts",0);

	//data.clear();
	data_vec.push_back(sdata);
	//data.initilize(length,xdata,ydata,polarity,tsdata);
	
	//sdata.clear();
	//engEvalString(ep,"clear all;"); //remove the data from matlab

}

/* process the current Spikedata object and push the spikes to matlab*/
void SpikeEnv::processl1spikes(void)
{
	int r_loc, c_loc, spk_event, l_max_r,l_max_c, l_min_r,l_min_c;
	uint t_ls,t_now;
	vector <int> l_c, l_r;

	//for saving spikes
	vector <udata> sp_x,sp_y,sp_p;
	vector <uint> sp_ts;
	int n_out_spikes = 0;

	int ncapture = 10;
	int interval = (data.nevents)/ncapture ;

	t_ls = 0;
	for(int e=0; e<data.nevents;e++)
	{
		l_c.clear(); 
		l_r.clear(); 

		c_loc = data.x[e]; 
		r_loc = data.y[e];
		spk_event = data.p[e]; //originally in the range 0,1
		t_now = data.ts[e];


		if (r_loc>=SENSOR_WIDTH_Y || c_loc > SENSOR_WIDTH_X)
			cout<< "Error in setting sensor width"<<endl;

		l_max_r = r_loc/params.STRIDE;
		l_max_r = (l_max_r>=nblocks_y)?nblocks_y-1:l_max_r;
		l_max_c = c_loc/params.STRIDE;
		l_max_c = (l_max_c>=nblocks_x)?nblocks_x-1:l_max_c;

		l_min_r = ceil(max(r_loc-params.DIM_BLOCK+1,0)/(float)params.STRIDE);
		l_min_c = ceil(max(c_loc-params.DIM_BLOCK+1,0)/(float)params.STRIDE);

		for(int i=l_min_r; i<l_max_r+1;i++)
			l_r.push_back(i);
		
		for(int i=l_min_c; i<l_max_c+1;i++)
			l_c.push_back(i);

		//should calculate for all combinations of l_c and l_r
		for (vector<int>::iterator it_r = l_r.begin() ; it_r != l_r.end(); ++it_r)
		{
			for (vector<int>::iterator it_c = l_c.begin() ; it_c != l_c.end(); ++it_c)
			{
				int r_idx, c_idx, r_blk_idx, c_blk_idx , blk_lin_idx, local_lin_idx;
				int pswitch;

				r_blk_idx = *it_r;
				c_blk_idx = *it_c;
				r_idx = r_loc - r_blk_idx*params.STRIDE;
				c_idx = c_loc - c_blk_idx*params.STRIDE;
				blk_lin_idx = c_blk_idx*nblocks_y + r_blk_idx;
				local_lin_idx = c_idx*params.DIM_BLOCK + r_idx;
				
				//pswitch = (spk_event>0)?0:params.DIM_BLOCK*params.DIM_BLOCK;
				pswitch = spk_event*params.DIM_BLOCK*params.DIM_BLOCK;
				
				syn_inputs.col(blk_lin_idx) *= exp(-params.LAMBDA_F*(t_now-t_ls));   
				syn_inputs(local_lin_idx+pswitch,blk_lin_idx)+=1;	

#if GETSPIKEHISTORY
				in_spike_count(blk_lin_idx,0)+=1;
				 
#endif
				if (sum(syn_inputs.col(blk_lin_idx)) > params.FIRETHRESH)
				{
					//cout<< "FIRE!\n ";
	
					Tmat X =  syn_inputs.col(blk_lin_idx);					
					syn_inputs.col(blk_lin_idx) *= params.RESETFAC;

					//call the encoder
					udata winner = enc.GASSOMEncode(X,blk_lin_idx,t_now,1);
					
					sp_x.push_back(r_blk_idx);
					sp_y.push_back(c_blk_idx);
					sp_p.push_back(winner);
					sp_ts.push_back(t_now);
					n_out_spikes++;

#if GETSPIKEHISTORY
					out_spike_count(blk_lin_idx,0)+=1;
					
#endif
				}
				

			}
		}//end of all instances;
		t_ls = t_now;
		
		if(((e)%interval)==0)
		{
			tomat::pushMat(enc.basis1,"basis1");
			tomat::pushMat(enc.basis2,"basis2");
			tomat::pushMat(out_spike_count,"spikes_out");
			tomat::pushMat(in_spike_count,"spikes_in");
			cout << float(e)/(data.nevents)*100<<endl;
		}
	}//end of event

	//push into matlab
	
	tomat::push(sp_x.data(), sp_x.size(), 1, "x", 0);
	tomat::push(sp_y.data(), sp_y.size(), 1, "y", 0);
	tomat::push(sp_p.data(), sp_p.size(), 1, "p", 0);
	tomat::push(sp_ts.data(), sp_ts.size(), 1, "ts", 0);
	tomat::push(&n_out_spikes, 1, 1, "nspikes", 0);
	
}

/* process the data_vec[ind] object and push the spikes to matlab
Used to process MNIST data
*/
void SpikeEnv::processl1spikes(int index)
{
	int r_loc, c_loc, spk_event, l_max_r,l_max_c, l_min_r,l_min_c;
	uint t_ls,t_now;
	vector <int> l_c, l_r;
	SpikeData sdata = data_vec[index];
	//for saving spikes
	vector <udata> sp_x,sp_y,sp_p;
	vector <uint> sp_ts;
	int n_out_spikes = 0;

	int ncapture = 10;
	int interval = (sdata.nevents)/ncapture ;

//	if (index==39)
//	{
//	cout<<30;
//	}
	enc.resetEncoder();
	t_ls = 0;
	for(int e=0; e<sdata.nevents;e++)
	{
		l_c.clear(); 
		l_r.clear(); 

		// R -> Y, C -> X
		c_loc = sdata.x[e]-1; //convert matlab 1 based indexing to 0 based 
		r_loc = sdata.y[e]-1;
		spk_event = sdata.p[e]-1; //event polarity is given as 2,1 in the data file 
		t_now = sdata.ts[e];

		if (r_loc>=SENSOR_WIDTH_Y || c_loc > SENSOR_WIDTH_X)
			cout<< "Error in setting sensor width"<<endl;

		l_max_r = r_loc/params.STRIDE;
		l_max_r = (l_max_r>=nblocks_y)?nblocks_y-1:l_max_r;
		l_max_c = c_loc/params.STRIDE;
		l_max_c = (l_max_c>=nblocks_x)?nblocks_x-1:l_max_c;

		l_min_r = ceil(max(r_loc-params.DIM_BLOCK+1,0)/(float)params.STRIDE);
		l_min_c = ceil(max(c_loc-params.DIM_BLOCK+1,0)/(float)params.STRIDE);

		for(int i=l_min_r; i<l_max_r+1;i++)
			l_r.push_back(i);
		
		for(int i=l_min_c; i<l_max_c+1;i++)
			l_c.push_back(i);

		//should calculate for all combinations of l_c and l_r
		for (vector<int>::iterator it_r = l_r.begin() ; it_r != l_r.end(); ++it_r)
		{
			for (vector<int>::iterator it_c = l_c.begin() ; it_c != l_c.end(); ++it_c)
			{
				int r_idx, c_idx, r_blk_idx, c_blk_idx , blk_lin_idx, local_lin_idx;
				int pswitch;

				r_blk_idx = *it_r;
				c_blk_idx = *it_c;
				r_idx = r_loc - r_blk_idx*params.STRIDE;
				c_idx = c_loc - c_blk_idx*params.STRIDE;
				blk_lin_idx = c_blk_idx*nblocks_y + r_blk_idx;
				local_lin_idx = c_idx*params.DIM_BLOCK + r_idx;
				
				pswitch = spk_event*params.DIM_BLOCK*params.DIM_BLOCK;

				syn_inputs.col(blk_lin_idx) *= exp(-params.LAMBDA_F*(t_now-t_ls));   
				syn_inputs(local_lin_idx+pswitch,blk_lin_idx)+=1;	

				

#if GETSPIKEHISTORY
				in_spike_count(blk_lin_idx,0)+=1;
#endif
				if (sum(syn_inputs.col(blk_lin_idx)) > params.FIRETHRESH)
				{
					//cout<< "FIRE!\n ";
	
					Tmat X =  syn_inputs.col(blk_lin_idx);					
					syn_inputs.col(blk_lin_idx) *= params.RESETFAC;

					//tomat::pushMat(X,"X");


					//call the encoder
					udata winner = enc.GASSOMEncode(X,blk_lin_idx,t_now,1);
					
					sp_x.push_back(c_blk_idx+1);
					sp_y.push_back(r_blk_idx+1);
					sp_p.push_back(winner+1);
					sp_ts.push_back(t_now);
					n_out_spikes++;

#if GETSPIKEHISTORY
					out_spike_count(blk_lin_idx,0)+=1;
#endif
				}
				

			}
		}//end of all instances;
		t_ls = t_now;
		
		if(((e)%interval)==0 && 0)
		{
			tomat::pushMat(enc.basis1,"basis1");
			tomat::pushMat(enc.basis2,"basis2");
			tomat::pushMat(out_spike_count,"spikes_out");
			tomat::pushMat(in_spike_count,"spikes_in");
			cout << float(e)/(sdata.nevents)*100<<endl;
		}
	}//end of event

	//push into matlab
	
	tomat::push(sp_x.data(), sp_x.size(), 1, "x", 0);
	tomat::push(sp_y.data(), sp_y.size(), 1, "y", 0);
	tomat::push(sp_p.data(), sp_p.size(), 1, "p", 0);
	tomat::push(sp_ts.data(), sp_ts.size(), 1, "ts", 0);
	tomat::push(&n_out_spikes, 1, 1, "nspikes", 0);

#if GETSPIKEHISTORY
	tomat::pushMat(out_spike_count,"spikes_out");
	tomat::pushMat(in_spike_count,"spikes_in");
#endif


#if DATA_IS_UCHAR == 1
	string s_x = "data{" + to_string(index+1) + "}{1} = uint8(x)";
	string s_y = "data{" + to_string(index+1) + "}{2} = uint8(y)";
	string s_p = "data{" + to_string(index+1) + "}{3} = uint8(p)";
	string s_t = "data{" + to_string(index+1) + "}{4} = uint32(ts)";
#else
	string s_x = "data{" + to_string(index+1) + "}{1} = uint16(x)";
	string s_y = "data{" + to_string(index+1) + "}{2} = uint16(y)";
	string s_p = "data{" + to_string(index+1) + "}{3} = uint16(p)";
	string s_t = "data{" + to_string(index+1) + "}{4} = uint32(ts)";
#endif

	engEvalString(ep, s_x.c_str());
	engEvalString(ep, s_y.c_str());
	engEvalString(ep, s_p.c_str());
	engEvalString(ep, s_t.c_str());

	
}

/*Use a portion of the training data itself as a test vector*/
float SpikeEnv::testLearning()
{
	int r_loc, c_loc, spk_event, l_max_r,l_max_c, l_min_r,l_min_c;
	uint t_ls,t_now;
	vector <int> l_c, l_r;
	int nout_spikes = 0;
	float proj_rerr = 0.0;

	int ntest_events = 1e6;
	ntest_events = ntest_events>data.nevents ? data.nevents:ntest_events;

		t_ls = 0;
		for(int e=0; e<ntest_events;e++)
		{
			l_c.clear(); 
			l_r.clear(); 

			r_loc = data.x[e]; 
			c_loc = data.y[e];
			spk_event = data.p[e];
			t_now = data.ts[e];

			l_max_r = r_loc/params.STRIDE;
			l_max_r = (l_max_r>=nblocks_y)?nblocks_y-1:l_max_r;
			l_max_c = c_loc/params.STRIDE;
			l_max_c = (l_max_c>=nblocks_x)?nblocks_x-1:l_max_c;

			l_min_r = ceil(max(r_loc-params.DIM_BLOCK+1,0)/(float)params.STRIDE);
			l_min_c = ceil(max(c_loc-params.DIM_BLOCK+1,0)/(float)params.STRIDE);

			for(int i=l_min_r; i<l_max_r+1;i++)
				l_r.push_back(i);
		
			for(int i=l_min_c; i<l_max_c+1;i++)
				l_c.push_back(i);

			//should calculate for all combinations of l_c and l_r
			for (vector<int>::iterator it_r = l_r.begin() ; it_r != l_r.end(); ++it_r)
			{
				for (vector<int>::iterator it_c = l_c.begin() ; it_c != l_c.end(); ++it_c)
				{
					int r_idx, c_idx, r_blk_idx, c_blk_idx , blk_lin_idx, local_lin_idx;
					int pswitch;

					r_blk_idx = *it_r;
					c_blk_idx = *it_c;
					r_idx = r_loc - r_blk_idx*params.STRIDE;
					c_idx = c_loc - c_blk_idx*params.STRIDE;
					blk_lin_idx = c_blk_idx*nblocks_x + r_blk_idx;
					local_lin_idx = c_idx*params.DIM_BLOCK + r_idx;
				
					pswitch = spk_event*params.DIM_BLOCK*params.DIM_BLOCK;

					syn_inputs.col(blk_lin_idx) *= exp(-params.LAMBDA_F*(t_now-t_ls));   
					syn_inputs(local_lin_idx+pswitch,blk_lin_idx)+=1;	


					if (sum(syn_inputs.col(blk_lin_idx)) > params.FIRETHRESH)
					{
						//cout<< "FIRE!\n ";
	
						Tmat X =  syn_inputs.col(blk_lin_idx);					
						syn_inputs.col(blk_lin_idx) *= params.RESETFAC;

						//tomat::pushMat(X,"X");

						//get the projection error
						proj_rerr += enc.projError(X);
						nout_spikes++;
					}
				

				}
			}//end of all instances;
			t_ls = t_now;
		

		}//end of event

		return proj_rerr/nout_spikes;
}


void SpikeEnv::learnTempotron(string data_file, Tempotron_params &tparams, int ndata, bool pool, bool validate)
{
	Tmat fire_data(enc.getparams().SUBNUM,ndata, fill::zeros); 
	string df =  "load('" + data_file + "');";
	
	//load the data
	engEvalString(ep,df.c_str());
		
	cout <<df<<endl;
	for(int im=0; im<ndata; im++ )
	{
		readMNISTDataPad(im, tparams.SPATIAL_RANGE[0]);
		cout << "reading data: "<< ((float)im)/ndata*100<<endl;
	}
	engEvalString(ep,"clear data;");


	//load the training labels
	string tlabs = "load('labels.mat');";
	engEvalString(ep,tlabs.c_str());
	int label=0;
	for(int i=0; i<ndata;i++)
	{
		string tlabs_load = "label=train_labels("+to_string(i+1)+");";
		engEvalString(ep,tlabs_load.c_str());
		tomat::get(&label,"label",0);
		data_labels.push_back(label);
	}
	//clean the matalb environment
	engEvalString(ep,"clear all;");

	tmpNetwork = new Tempotron(tparams,data_vec,data_labels );
	tmpNetwork->train(pool, validate);

	delete tmpNetwork;
}

void SpikeEnv::testTempotron(string data_file, string weights_file, Tempotron_params &tparams, int ndata, bool pool)
{
	Tmat fire_data(enc.getparams().SUBNUM,ndata, fill::zeros); 
	string df =  "load('" + data_file + "');";
	
	//load the data
	engEvalString(ep,df.c_str());
		
	cout <<df<<endl;
	for(int im=0; im<ndata; im++ )
	{
		readMNISTDataPad(im, tparams.SPATIAL_RANGE[0]);
		cout << "reading data: "<< ((float)im)/ndata*100<<endl;
	}
	engEvalString(ep,"clear data;");


	//load the training labels
	string tlabs = "load('labels.mat');";
	engEvalString(ep,tlabs.c_str());
	int label=0;
	for(int i=0; i<ndata;i++)
	{
		string tlabs_load = "label=test_labels("+to_string(i+1)+");";
		engEvalString(ep,tlabs_load.c_str());
		tomat::get(&label,"label",0);
		data_labels.push_back(label);
	}

	string wf =  "load('" + weights_file + "');";
	engEvalString(ep,wf.c_str());

	Tmat weights = Tmat(tparams.N,tparams.NUM_NEURONS, fill::zeros);
	tomat::getMat(weights,"weights");
	//tomat::pushMat(weights,"weights_pushed");
	//clean the matalb environment
	engEvalString(ep,"clear all;");

	tmpNetwork = new Tempotron(tparams,data_vec,data_labels );
	tmpNetwork->loadWeights(weights);
	tmpNetwork->test(0,false);
	
	engEvalString(ep,"filename = datestr(now, 'yyyymmdd_HHMMSS');\
		save(['data/',filename,'_Tempotron_test.mat']);");

	delete tmpNetwork;
}


void SpikeEnv::saveDatatoMat()
{
	//export the gassom and environment parameters and the learned basis vectors
	int STRIDE = params.STRIDE;
	int DIM_BLOCK = params.DIM_BLOCK;
	int SENSOR_WIDTH_X = params.SENSOR_WIDTH_X;
	int SENSOR_WIDTH_Y = params.SENSOR_WIDTH_Y;
	float RESETFAC = params.RESETFAC;
	int NTIMES = params.NTIMES;
	float LAMBDA_F = params.LAMBDA_F;
	float FIRETHRESH = params.FIRETHRESH;

	int TOPO_SUBSPACE = enc.getparams().TOPO_SUBSPACE;
	int SUBNUM = enc.getparams().SUBNUM;
	int NBPERSUB = enc.getparams().NBPERSUB;
	int BASISDIM = enc.getparams().BASISDIM;
	int BATCH_SIZE = enc.getparams().BATCH_SIZE;
	float ALPHA_A = enc.getparams().ALPHA_A;
	float ALPHA_C = enc.getparams().ALPHA_C;
	float SIGMA_A = enc.getparams().SIGMA_A; 
	float SIGMA_C = enc.getparams().SIGMA_C;
	float SIGMA_N = enc.getparams().SIGMA_N;
	float SIGMA_W = enc.getparams().SIGMA_W;
	float TCONST = enc.getparams().TCONST;
	float SLOWNESS_TRANS = enc.getparams().SLOWNESS_TRANS;
	float LAMBDA_SAC = enc.getparams().LAMBDA_SAC; 
	int UPDATE_FREQ = enc.getparams().UPDATE_FREQ;

	tomat::push(&STRIDE,1,1,"stride",0);
	tomat::push(&DIM_BLOCK,1,1,"dim_block",0);
	tomat::push(&SENSOR_WIDTH_X,1,1,"sensor_width_x",0);
	tomat::push(&SENSOR_WIDTH_Y,1,1,"sensor_width_y",0);
	tomat::push(&LAMBDA_F,1,1,"lambda_frame",0);
	tomat::push(&FIRETHRESH,1,1,"fire_threshold",0);

	tomat::push(&TOPO_SUBSPACE,1,1,"topo_subspace",0);
	tomat::push(&BASISDIM,1,1,"basis_dim",0);
	tomat::push(&BATCH_SIZE,1,1,"batch_size",0);
	tomat::push(&ALPHA_A,1,1,"alpha_a",0);
	tomat::push(&ALPHA_C,1,1,"alpha_c",0);
	tomat::push(&SIGMA_A,1,1,"sigma_a",0);
	tomat::push(&SIGMA_C,1,1,"sigma_c",0);
	tomat::push(&TCONST,1,1,"tconst",0);
	tomat::push(&SLOWNESS_TRANS,1,1,"rho",0);
	tomat::push(&ALPHA_A,1,1,"batch_size",0);
	tomat::push(&LAMBDA_SAC,1,1,"lambda_sac",0);
	tomat::push(&ALPHA_A,1,1,"batch_size",0);

	tomat::pushMat(enc.basis1,"basis1");
	tomat::pushMat(enc.basis2,"basis2");

	engEvalString(ep,"filename = datestr(now, 'yyyymmdd_HHMMSS');\
		save(['data/',filename,'.mat']);");

}