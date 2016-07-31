#pragma once

#include "configs.h"

class SpikeData
{
public:	
	int nevents;
	udata *x,*y,*p;
	uint *ts;

public:
	void initilize(int nevents, udata *x, udata*y, udata *p, uint *ts );
	void clear(void);
	~SpikeData();
	SpikeData();
	SpikeData(const SpikeData &spkdata);
};