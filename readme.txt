Spiking GASSOM and tempetron based classification implementation 
tnc<at>connect<dot>ust<dot>hk
Publication : http://arxiv.org/abs/1604.04327 <temporary link> Published in BIOROB 2016, held in Singapore. 
Free to be used in Academic work. 

Please note that this is a crude version of research code, and not intended to be used as a blackbox.  

Required Libraries
	1) Armidillo -> http://arma.sourceforge.net/
	2) Matlab engine
	3) Visual studio 2012 
Set up the correct paths for finding the headers and linked libraries. Should be straightforward based on the values I used for my PC. 

Samples and essential files neeeded to run the code:
	The required data files and matlab snippets are in a dropbox folder. It can be downloaded from -> https://www.dropbox.com/sh/avhshvhkk0kc1hl/AAAR6Jai4FUDNk-s3KdAS7-pa?dl=0
	You will have to point to the folder location in the main() function of the spiking_main.cpp. 
		
The code can do the following. 
	1) Learn a representation from an input event stream (from a DVS sensor) using the event based GASSOM framework
	2) Used a learn set of GASSOM feature detectors to encode an event stream from a DVS sensor   
	3) Use the tempotron network to learn a supervised classifier
	4) Use the learned tempotron network to perform classification
	
	
Main classes
	1) SpikeEnv Class: Used in training the GASSOM. Implements the leaky integrators. Containes a GASSOM object whch is called when an output event is generated.
	2) GASSOM_Spk Class: Carries out the learning and encoding part for output events generated using the SpikeEnv object. 
	3) Tempotron Class: Supervised classification on an output event set generated using the event based GASSOM encoder. 
	
Main use cases: 
	1) Learning a GASSOM based representation:
			# 	Select the correct configuration in the configs.h file. Use the #if -- #endif sections to activate the correct settings. Change any settings if needed.
			#	in the main() function of the spiking_main.cpp.  -> Make sure the active matlab work path is selected by setting the 'matlab_work_directory' Directory.  Uncomment -> trainL1basis()  and comment others. 
			#	Make sure trainL1basis() in spiking_main.cpp has the correct data file name specified. For the format of the datafile, check the sample supplied. 
			#   The trained basis will be saved to the matlab context. Make sure a 'data' folder is present at the active matlab path.  	
	
	2) Encoding a saved data stream with learned GASSOM basis vectors
			#  	Select the correct configuration in the configs.h file. Use the #if -- #endif sections to activate the correct settings (the /* For L1 basis training*/ or /* MNIST get L1 spikes*/). Change any settings if needed.
			#	in the main() function of the spiking_main.cpp.  -> Make sure the active matlab work path is selected by setting the 'matlab_work_directory' Directory.  Uncomment -> getL1Spikes()  and comment others. 
			#	In the getL1Spikes() function enter the 'saved_bases_file' and 'input_events_file' variables. 
			#   After processing the output events are stored in x,y,p,ts,nspikes fields in matlab. Save them to a place you like.  
			#	Note that MNIST output spikes are processed using getMNISTSpikes() function. You will have to edit 'data_file' 'save_file' locations correctly.   So basically you might have to write customized function for your work. 
	
	3) Using the Tempotron to learn event based supervised classification (Learning the network)
			#	We assume the output events are saved in the prescribed format. Check the samples for the format. 
			#   In the main() function of the spiking_main.cpp.  -> Make sure the active matlab work path is selected by setting the 'matlab_work_directory' Directory.  Uncomment -> tempotronClassify()  and comment others. 
			#	In tempotronClassify() The 'data_file' parameter should be set
	
	4)	Using a learned tempotron network to classify a test set
			# use the tempotronClassifyTest()  function and set the parameters 