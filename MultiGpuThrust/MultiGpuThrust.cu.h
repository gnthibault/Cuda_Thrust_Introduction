/*
 * MultiGpuThrust.cu.h
 *
 *  Created on: 12 juin 2015
 *      Author: Thibault Notargiacomo
 */

#ifndef MULTIGPUTHRUST_CU_H_
#define MULTIGPUTHRUST_CU_H_

//STL
#include <vector>
#include <algorithm>
#include <functional>
#include <cmath>
#include <memory>

//Thrust
#include <thrust/device_vector.h>

void reduceVector( std::vector<std::shared_ptr<thrust::device_vector<int> > >& v, int tid, int maxTid )
{
	if( tid + maxTid <  v.size() )
	{
		//Set current device
		cudaSetDevice( tid );

		// We add vector tid and vector tid+maxTid and put the result into vector tid
		thrust::transform( v.at(tid)->begin(), v.at(tid)->end(), v.at(tid+maxTid)->begin(), v.at(tid)->begin(), thrust::plus<int>() );
	}
}

int giveReductionSize( int in )
{
	//If number is a power of two, then returns the number divided by 2
	//Else give the next power of two fthat follows the number divided by two
	if( in > 0 )
	{
		return std::pow(2, std::ceil(std::log2(in/2.0)) );
	}else
	{
		return 0;
	}
}


#endif /* MULTIGPUTHRUST_CU_H_ */
