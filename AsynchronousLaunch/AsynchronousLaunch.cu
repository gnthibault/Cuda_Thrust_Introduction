/*
 * AsynchronousLaunch.cu
 *
 *  Created on: 12 juin 2015
 *      Author: Thibault Notargiacomo
 */


//STL
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <functional>
#include <vector>

//Thrust
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

//Cuda
#include <cuda_runtime.h>

//Local
#include "AsynchronousLaunch.cu.h"

int main( int argc, char* argv[] )
{
	const size_t fullSize =  1024*1024*2;
	const size_t nbOfStrip = 4;
	const size_t stripSize =  fullSize/nbOfStrip;

	//Allocate host pinned memory in order to use asynchronous api and initialize it with random values
	float* hostVector;
	cudaMallocHost(&hostVector,fullSize*sizeof(float));
	std::generate(hostVector, hostVector+fullSize, normalRandomFunctor<float>(0.f,1.f) );

	//And one device vector of the same size
	thrust::device_vector<float> deviceVector( fullSize );
	thrust::device_vector<float> deviceVector2( fullSize );

	//Declare  and initialize also two cuda stream
	std::vector<cudaStream_t> vStream(nbOfStrip);
	for( auto it = vStream.begin(); it != vStream.end(); it++ )
	{
		cudaStreamCreate( &(*it) );
	}

	//Now, we would like to perform an alternate scheme copy/compute in a loop using the copyToDevice/Compute/CopyToHost for each stream scheme:
	for( int i = 0; i < 5; i++ )
	{
		for( int j=0; j!=nbOfStrip; j++)
		{
			size_t offset = stripSize*j;
			size_t nextOffset = stripSize*(j+1);
			cudaStreamSynchronize(vStream.at(j));
			cudaMemcpyAsync(thrust::raw_pointer_cast(deviceVector.data())+offset, hostVector+offset, stripSize*sizeof(float), cudaMemcpyHostToDevice, vStream.at(j));
			thrust::transform( thrust::cuda::par.on(vStream.at(j)), deviceVector.begin()+offset, deviceVector.begin()+nextOffset, deviceVector.begin()+offset, computeFunctor<float>() );
			cudaMemcpyAsync(hostVector+offset, thrust::raw_pointer_cast(deviceVector.data())+offset, stripSize*sizeof(float), cudaMemcpyDeviceToHost, vStream.at(j));
		}
	}
	//On devices that do not possess multiple queues copy engine capability, this solution serializes all command even if they have been issued to different streams
	//Why ? Because in the point of view of the copy engine, which is a single ressource in this case, there is a time dependency between HtoD(n) and DtoH(n) which is ok, but there is also
	// a false dependency between DtoH(n) and HtoD(n+1), that preclude any copy/compute overlap

	//Full Synchronize before testing second solution
	cudaDeviceSynchronize();

	//Now, we would like to perform an alternate scheme copy/compute in a loop using the copyToDevice for each stream /Compute for each stream /CopyToHost for each stream scheme:
	for( int i = 0; i < 5; i++ )
	{
		for( int j=0; j!=nbOfStrip; j++)
		{
			cudaStreamSynchronize(vStream.at(j));
		}
		for( int j=0; j!=nbOfStrip; j++)
		{
			size_t offset = stripSize*j;
			cudaMemcpyAsync(thrust::raw_pointer_cast(deviceVector.data())+offset, hostVector+offset, stripSize*sizeof(float), cudaMemcpyHostToDevice, vStream.at(j));
		}
		for( int j=0; j!=nbOfStrip; j++)
		{
			size_t offset = stripSize*j;
			size_t nextOffset = stripSize*(j+1);
			thrust::transform( thrust::cuda::par.on(vStream.at(j)), deviceVector.begin()+offset, deviceVector.begin()+nextOffset, deviceVector.begin()+offset, computeFunctor<float>() );

		}
		for( int j=0; j!=nbOfStrip; j++)
		{
			size_t offset = stripSize*j;
			cudaMemcpyAsync(hostVector+offset, thrust::raw_pointer_cast(deviceVector.data())+offset, stripSize*sizeof(float), cudaMemcpyDeviceToHost, vStream.at(j));
		}
	}
	//On device that do not possess multiple queues in the copy engine, this solution yield better results, on other, it should show nearly identic results

	//Full Synchronize before exit
	cudaDeviceSynchronize();

	for( auto it = vStream.begin(); it != vStream.end(); it++ )
	{
		cudaStreamDestroy( *it );
	}
	cudaFreeHost( hostVector );

	return EXIT_SUCCESS;
}
