/*
 * MultiGpuThrust.cu
 *
 *  Created on: 12 juin 2015
 *      Author: Thibault Notargiacomo
 *
 * Content: This code intends to show how it is possible to reduce multiple vector
 * each located on a different gpu, towards a single vector, located on the first GPU
 * any commutative operator can be used, we have chosen addition here
 */

//STL
#include <iostream>
#include <memory>

//Cuda
#include "cuda_runtime.h"

//Local
#include "MultiGpuThrust.cu.h"
#include "../Include/cudaHelper.cu.h"

int main( int argc, char* argv[] )
{
	//Size parameters
	int sizeVector = 10;

	//vector of pointer to thrust::device_vector
	std::vector<std::shared_ptr<thrust::device_vector<int> > > vpDeviceVector;

	//Look for all devices on the current platform
	int nb_device = 0;
	checkCudaErrors( cudaGetDeviceCount( &nb_device ) );

	// initialize vector for all available GPU, if peer to peer acces is impossible, this step may fail,
	// see cudaDeviceCanAccessPeer to get something more robust
	for( int i = 0; i != nb_device; i++ )
	{
		//Set device as the current device
		checkCudaErrors( cudaSetDevice( i ) );

		//Initialize memory
		vpDeviceVector.emplace_back( std::make_shared<thrust::device_vector<int> >( sizeVector, 111 ) );

		//Enable Peer to Peer access, ie, current device can acces to memory of all superior device IDs
		for( int j = i+1; j < nb_device; j++ )
		{
			checkCudaErrors( cudaDeviceEnablePeerAccess(j, 0) );
		}
	}
	// This is where reduction take place
	int maxTid = giveReductionSize(nb_device);
	while( maxTid != 0 )
	{
		//Launch a group of threads
		for(int i = 0; i < maxTid; ++i)
		{
			reduceVector( vpDeviceVector, i, maxTid );
		}
		//Half the work is remaining
		maxTid /= 2;
	}
	//Check if the sum is correct
	for(int i = 0; i < sizeVector; ++i)
	{
		std::cout << " The sum is " << (*vpDeviceVector.at(0))[i] << " (should be " << nb_device*111 << " )" << std::endl;
	}
	return EXIT_SUCCESS;
}
