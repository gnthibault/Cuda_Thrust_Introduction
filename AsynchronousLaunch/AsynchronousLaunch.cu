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
#include <vector>
#include <functional>

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
	const size_t fullSize = 1024*1024*16;
	const size_t halfSize = fullSize/2;

	//Declare one host std::vector and initialize it with random values
	std::vector<float> hostVector( fullSize );
	std::generate(hostVector.begin(), hostVector.end(), normalRandomFunctor<float>(0.f,1.f) );

	//And two device vector of Half size
	thrust::device_vector<float> deviceVector0( halfSize );
	thrust::device_vector<float> deviceVector1( halfSize );

	//Declare  and initialize also two cuda stream
	cudaStream_t stream0, stream1;
	cudaStreamCreate( &stream0 );
	cudaStreamCreate( &stream1 );

	//Now, we would like to perform an alternate scheme copy/compute
	for( int i = 0; i < 10; i++ )
	{
		//Wait for the end of the copy to host before starting to copy back to device
		cudaStreamSynchronize(stream0);
		//Warning: thrust::copy does not handle asynchronous behaviour for host/device copy, you must use cudaMemcpyAsync to do so
		cudaMemcpyAsync(thrust::raw_pointer_cast(deviceVector0.data()), thrust::raw_pointer_cast(hostVector.data()), halfSize*sizeof(float), cudaMemcpyHostToDevice, stream0);
		cudaStreamSynchronize(stream1);
		//second copy is most likely to occur sequentially after the first one
		cudaMemcpyAsync(thrust::raw_pointer_cast(deviceVector1.data()), thrust::raw_pointer_cast(hostVector.data())+halfSize, halfSize*sizeof(float), cudaMemcpyHostToDevice, stream1);

		//Compute on device, here inclusive scan, for histogram equalization for instance
		thrust::transform( thrust::cuda::par.on(stream0), deviceVector0.begin(), deviceVector0.end(), deviceVector0.begin(), computeFunctor<float>() );
		thrust::transform( thrust::cuda::par.on(stream1), deviceVector1.begin(), deviceVector1.end(), deviceVector1.begin(), computeFunctor<float>() );

		//Copy back to host
		cudaMemcpyAsync(thrust::raw_pointer_cast(hostVector.data()), thrust::raw_pointer_cast(deviceVector0.data()), halfSize*sizeof(float), cudaMemcpyDeviceToHost, stream0);
		cudaMemcpyAsync(thrust::raw_pointer_cast(hostVector.data())+halfSize, thrust::raw_pointer_cast(deviceVector1.data()), halfSize*sizeof(float), cudaMemcpyDeviceToHost, stream1);
	}

	//Full Synchronize before exit
	cudaDeviceSynchronize();

	cudaStreamDestroy( stream0 );
	cudaStreamDestroy( stream1 );

	return EXIT_SUCCESS;
}
