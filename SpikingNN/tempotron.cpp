#include "tempotron.h"

#include "toMatlab.h"

Tempotron::~Tempotron(){}

Tempotron::Tempotron(Tempotron_params pr,vector<SpikeData>& dv, vector<int> &lv):
	params(pr), data_vec(dv), data_labels(lv)
{
	weights = Tmat(params.N,params.NUM_NEURONS,fill::randn)*0.01;
	grads =   Tmat(params.N,params.NUM_NEURONS,fill::zeros);

	//initilize the delay_to_max time and V0
	float T = params.TAU1*5; 
	int nsteps = int(T/params.dt);
	float maxval=0, v1=0, v2=0, tmax=0;
	for(int i=0;i<nsteps;i++)
	{
		float val = exp(-params.dt*i/params.TAU1)-exp(-params.dt*i/params.TAU2);
		if(val>maxval)
		{
			maxval = val;
			tmax = params.dt*i;
		}
	}
	params.delay_to_max = tmax; 
	params.V0 = 1/maxval; 



}



void Tempotron::train(bool pool, bool validate)
{
	float max_train_accuracy = 0;
	Tmat cofusion;
	int n_train_samples;
	if(validate)
	{
		n_train_samples = (1-params.val_ratio)*data_labels.size();
	}
	else
	{
		n_train_samples = data_labels.size();
	}
	grads_prev = Tmat(params.N,params.NUM_NEURONS,fill::zeros);
	
	for(int rep=0; rep<params.iterations;rep++)
	{
		int nbatches = floor(n_train_samples/params.batch_size);
		int correct_count =0;
		params.LRATE *= 0.9995; //reduce learning rate 
		cofusion = Tmat(params.NUM_NEURONS,params.NUM_NEURONS+1,fill::zeros);
		for(int bt=0; bt<nbatches; bt++)
		{
			int it_start = bt*params.batch_size;
			correct_count = 0;
			grads =   Tmat(params.N,params.NUM_NEURONS,fill::zeros);
			 
			
			//this loop has to be parallalized. Each spikedata item has to be processed independently, except the common matrix for gradiants
			for(int it=0; it<params.batch_size; it++) // iterate through the items
			{
				// most of these objects are localized to make parallel processing possible
				Tmat spvec1 =  Tmat(params.N,1,fill::zeros);
				Tmat spvec2 =  Tmat(params.N,1,fill::zeros);
				Tmat K_ti_max =Tmat(params.N,params.NUM_NEURONS,fill::zeros); // for non firing, to store the Ki at tmax
				float tlast = 0;
				Tmat maxV = Tmat(1,params.NUM_NEURONS,fill::ones)*-1;
				Tmat maxT = Tmat(1,params.NUM_NEURONS,fill::zeros); 
				Tmat update_vec = Tmat(1,params.NUM_NEURONS,fill::zeros); // 0:No change, i.e. correct update; 1:positive update,  i.e did not fire correctly; -1:negative update, i.e. fired incorrectly
				queue<float> triggerQ;
				int idx = it+it_start;
				SpikeData item = data_vec.at(idx);
				int label = data_labels.at(idx);
				int triggerTrack = params.NUM_NEURONS; // this is used to check which neuron fired first. default is no fire. 
				bool isTriggered = false; 
				bool correct_class = true; 
				bool spiked[10] = {false};
				
				for(int e=0; e<item.nevents; e++) // go through the events one by one
				{//do processing
					int afferent_idx;
					if(pool)
						afferent_idx = item.p[e]-1 ;
					else
						afferent_idx = item.p[e]-1 + params.EVENT_TYPES*( (item.x[e]-1)*params.SPATIAL_RANGE[0]+ item.y[e]-1 ) ;

					float tnow = item.ts[e];
					
					while(!triggerQ.empty() && triggerQ.front() < tnow )
					{
						float spktime = triggerQ.front();
						triggerQ.pop();
						Tmat e1tmp = spvec1*exp(-(spktime-tlast)/params.TAU1);
						Tmat e2tmp = spvec2*exp(-(spktime-tlast)/params.TAU2);
						Tmat K_i = e1tmp - e2tmp;
						//tomat::pushMat(e1tmp,"e1tmp");
						//tomat::pushMat(e2tmp,"e2tmp");
						//tomat::pushMat(spvec1,"spvec1");
						//tomat::pushMat(spvec2,"spvec2");

						//process all neurons
						for(int ni=0; ni<params.NUM_NEURONS; ni++)
						{
							if(spiked[ni]==false) //if the neuron has not spiked for the pattern
							{
								float V = accu(K_i%weights.col(ni))+params.VR;
								if(V>=params.VT)
								{
									maxV(ni) = V;
									maxT(ni) = spktime;
									K_ti_max.col(ni) = K_i;
									isTriggered = true;
									triggerTrack = ni;
									spiked[ni] = true;
									if(label!=ni) //wrong neuron fired
									{
										update_vec(ni) = -1;
										correct_class = false;
									}

									break; 
								}
								else //no firing, but still have to update the grad if the V is highest
								{
									if(V>maxV(ni))
									{
										K_ti_max.col(ni) = K_i;
										maxV(ni) = V;
										maxT(ni) = spktime;
									}
								}
							}
						}


					} // end of trigger checking

					if(isTriggered) //experimental. Only one neuron gets to fire. 
					{
						break;
					}

					//update the surfaces
					spvec1 *= exp(-(tnow-tlast)/params.TAU1);
					spvec2 *= exp(-(tnow-tlast)/params.TAU2);
					//tomat::pushMat(spvec1,"spvec1");

					spvec1(afferent_idx)+=params.V0; // 1 based indexing for event type 
					spvec2(afferent_idx)+=params.V0;

					//tomat::pushMat(spvec1,"spvec1");
					
					//if(weights(item.p[e]-1)>0) // for positive weights push to the Q. 
					{
						triggerQ.push(tnow+params.delay_to_max);
					}
					tlast = tnow; 
					
				}// end of event streaming


				if(!isTriggered) //if the correct neuron has not fired, update it with +1 weight
				{
					update_vec(label) = 1; //label has 0 based indexing, no neuron has fired. So the correct weights should be positively updated. 
					correct_class = false;
				}

				if (correct_class)
					correct_count++;
				cofusion(label,triggerTrack)+=1; // 100.0/float(data_labels.size());

				// update the gradiant. Critical section
				Tmat update_mat = repmat(update_vec,params.N,1); //[N_aff, N_Neurons]
				Tmat update_vals = params.LRATE*(K_ti_max%update_mat);
//				tomat::pushMat(update_mat,"update_mat");
//				tomat::pushMat(K_ti_max,"K_ti_max");
//				tomat::pushMat(update_vals,"update_vals");
//				tomat::pushMat(spvec1,"spvec1");
//				tomat::pushMat(spvec2,"spvec2");
//				tomat::pushMat(weights,"weights");
//				tomat::pushMat(maxV,"maxV");
//				tomat::pushMat(maxT,"maxT");
				grads += update_vals;

				//cout<<"Percentage completed: "<< float(idx+rep*nbatches*params.batch_size)/float(nbatches*params.batch_size*params.iterations)*100<< endl;


			}// end of mini batch
			//tomat::pushMat(weights,"weights_pre");
			grads*= 1.0/float(params.batch_size); //normalizing to remove batch size from effecting the update rate
			grads+= params.MRATE*grads_prev; //momentum
			grads_prev = grads;
			weights = weights + grads; //weight update
			tomat::pushMat(weights,"weights");
			//tomat::pushMat(grads,"grads");
			
			float batch_train_accuracy = float(correct_count)/float(params.batch_size)*100;
			if(batch_train_accuracy>max_train_accuracy)
				max_train_accuracy = batch_train_accuracy;
			cout<< "\t***" << "Correct Classification: "<<  batch_train_accuracy<< endl;
			cout<<"Percentage completed: "<< float(bt*params.batch_size+rep*nbatches*params.batch_size)/float(nbatches*params.batch_size*params.iterations)*100<< endl;
			
		}
		Tmat sums = repmat(sum(cofusion,1),1,params.NUM_NEURONS+1)/100.0;
		tomat::pushMat(sums,"sums_train");
		cofusion /= sums;
		tomat::pushMat(cofusion,"confusion_train");
				
	}
	tomat::push(&max_train_accuracy,1,1,"max_train_accuracy");

	//start validations
	if(validate)
		test(n_train_samples,pool);
	saveWeights();
	cout<<"End of training and validation..."<<endl;
}


void Tempotron::test(int start_idx,bool pool)
{
	cout<< "starting testing/validation"<<endl;
	Tmat cofusion = Tmat(params.NUM_NEURONS,params.NUM_NEURONS+1,fill::zeros);
	int correct_count = 0;
	for(int it=start_idx; it<data_labels.size(); it++) // iterate through the items
			{
				// most of these objects are localized to make parallel processing possible
				Tmat spvec1 =  Tmat(params.N,1,fill::zeros);
				Tmat spvec2 =  Tmat(params.N,1,fill::zeros);
				Tmat K_ti_max =Tmat(params.N,params.NUM_NEURONS,fill::zeros); // for non firing, to store the Ki at tmax
				float tlast = 0;
				Tmat maxV = Tmat(1,params.NUM_NEURONS,fill::ones)*-1;
				Tmat maxT = Tmat(1,params.NUM_NEURONS,fill::zeros); 
				Tmat update_vec = Tmat(1,params.NUM_NEURONS,fill::zeros); // 0:No change, i.e. correct update; 1:positive update,  i.e did not fire correctly; -1:negative update, i.e. fired incorrectly
				queue<float> triggerQ;
				int idx = it;
				SpikeData item = data_vec.at(idx);
				int label = data_labels.at(idx);
				int triggerTrack = params.NUM_NEURONS; // this is used to check which neuron fired first. default is no fire. 
				bool isTriggered = false; 
				bool correct_class = true; 
				bool spiked[10] = {false};
				
				for(int e=0; e<item.nevents; e++) // go through the events one by one
				{//do processing
					int afferent_idx;
					if(pool)
						afferent_idx = item.p[e]-1 ;
					else
						afferent_idx = item.p[e]-1 + params.EVENT_TYPES*( (item.x[e]-1)*params.SPATIAL_RANGE[0]+ item.y[e]-1 ) ;

					float tnow = item.ts[e];
					
					while(!triggerQ.empty() && triggerQ.front() < tnow )
					{
						float spktime = triggerQ.front();
						triggerQ.pop();
						Tmat e1tmp = spvec1*exp(-(spktime-tlast)/params.TAU1);
						Tmat e2tmp = spvec2*exp(-(spktime-tlast)/params.TAU2);
						Tmat K_i = e1tmp - e2tmp;
						//tomat::pushMat(e1tmp,"e1tmp");
						//tomat::pushMat(e2tmp,"e2tmp");
						//tomat::pushMat(spvec1,"spvec1");
						//tomat::pushMat(spvec2,"spvec2");

						//process all neurons
						for(int ni=0; ni<params.NUM_NEURONS; ni++)
						{
							if(spiked[ni]==false) //if the neuron has not spiked for the pattern
							{
								float V = accu(K_i%weights.col(ni))+params.VR;
								if(V>=params.VT)
								{
									maxV(ni) = V;
									maxT(ni) = spktime;
									K_ti_max.col(ni) = K_i;
									isTriggered = true;
									triggerTrack = ni;
									spiked[ni] = true;
									if(label!=ni) //wrong neuron fired
									{
										correct_class = false;
									}

									break; 
								}
							}
						}


					} // end of trigger checking

					if(isTriggered) //experimental. Only one neuron gets to fire. 
					{
						break;
					}

					//update the surfaces
					spvec1 *= exp(-(tnow-tlast)/params.TAU1);
					spvec2 *= exp(-(tnow-tlast)/params.TAU2);
					//tomat::pushMat(spvec1,"spvec1");

					spvec1(afferent_idx)+=params.V0; // 1 based indexing for event type 
					spvec2(afferent_idx)+=params.V0;

					//tomat::pushMat(spvec1,"spvec1");
					
					//if(weights(item.p[e]-1)>0) // for positive weights push to the Q. 
					{
						triggerQ.push(tnow+params.delay_to_max);
					}
					tlast = tnow; 
					
				}// end of event streaming


				if(!isTriggered) //if the correct neuron has not fired, update it with +1 weight
				{
					correct_class = false;
				}

				if (correct_class)
					correct_count++;
				cofusion(label,triggerTrack)+=1; // 100.0/float(data_labels.size());

				cout<<"Percentage completed: "<< float(it - start_idx)/float(data_labels.size()-start_idx)*100<< endl;

			}// end samples
			//tomat::pushMat(weights,"weights_pre");
			
			float test_accuracy = float(correct_count)/float(data_labels.size()-start_idx)*100;
			cout<< "\t***" << "Correct Test Classification: "<<  test_accuracy<< endl;

			Tmat sums = repmat(sum(cofusion,1),1,params.NUM_NEURONS+1)/100.0;
			tomat::pushMat(sums,"sums_test");
			cofusion /= sums;
			tomat::pushMat(cofusion,"confusion_test");
			

}


void Tempotron::loadWeights(Tmat &wt)
{
	weights = wt; 
}

void Tempotron::saveWeights()
{
	tomat::push(&params.LRATE,1,1,"lrate",0);
	tomat::push(&params.TAU1,1,1,"tau1",0);
	tomat::push(&params.TAU2,1,1,"tau2",0);
	tomat::push(&params.VT,1,1,"VT",0);
	tomat::push(&params.VR,1,1,"VR",0);
	tomat::push(&params.N,1,1,"N",0);
	tomat::push(&params.NUM_NEURONS,1,1,"NUM_NEURONS",0);
	tomat::push(&params.dt,1,1,"dt",0);
	tomat::push(&params.batch_size,1,1,"batch_size",0);
	tomat::push(&params.iterations,1,1,"iterations",0);
	tomat::push(&params.delay_to_max,1,1,"delay_to_max",0);
	tomat::push(&params.V0,1,1,"V0",0);
	tomat::pushMat(weights,"weights");
	engEvalString(ep,"filename = datestr(now, 'yyyymmdd_HHMMSS');\
		save(['data/',filename,'_Tempotron_train.mat']);");
}