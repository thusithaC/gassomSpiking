#include "spikeData.h"

void SpikeData::initilize(int len, udata *xd, udata*yd, udata *pd, uint *tsd )
{
	
	nevents = len;
	x = xd;
	y = yd;
	p = pd;
	ts = tsd;
}

 SpikeData::SpikeData(void)
{
	x = NULL;
	y = NULL;
	p = NULL;
	ts = NULL;
	nevents = 0;

}

 SpikeData::SpikeData(const SpikeData &spkdata)
{

	nevents = spkdata.nevents;
	x = new udata[nevents];
	y = new udata[nevents];
	p = new udata[nevents];
	ts = new uint[nevents];

	memcpy(x, spkdata.x, sizeof(udata)*nevents);
	memcpy(y, spkdata.y, sizeof(udata)*nevents);
	memcpy(p, spkdata.p, sizeof(udata)*nevents);
	memcpy(ts, spkdata.ts, sizeof(uint)*nevents);

	

}
 SpikeData::~SpikeData(void)
 {
	
	 if (nevents>0)
	 {
		 delete[] x;
		 delete[] y;
		 delete[] p;
		 delete[] ts;
		 x = NULL;
		 y = NULL;
		 p = NULL;
		 ts = NULL;
		 nevents = 0; 
	 }
 }

void SpikeData::clear(void)
 {	
	 if (nevents>0)
	 {
		 delete[] x;
		 delete[] y;
		 delete[] p;
		 delete[] ts;
		 x = NULL;
		 y = NULL;
		 p = NULL;
		 ts = NULL;
		 nevents = 0; 
	 }
 }
