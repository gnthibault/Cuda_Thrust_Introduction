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

//Thrust
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>

//Cuda
#include <cuda_runtime.h>

//Local
#include "AsynchronousLaunch.cu.h"

int main( int argc, char* argv[] )
{
	const size_t fullSize = 1024*1024*512;
	const size_t halfSize = fullSize/2;

	//Declare one host std::vector and initialize it with random values
	std::vector<int> hostVector( fullSize );
	std::generate(hostVector.begin(), hostVector.end(), std::rand);

	//And two device vector of Half size
	thrust::device_vector<int> deviceVector0( halfSize );
	thrust::device_vector<int> deviceVector1( halfSize );

	//Declare  and initialize also two cuda stream
	cudaStream_t stream0, stream1;
	cudaStreamCreate( &stream0 );
	cudaStreamCreate( &stream1 );

	//Now, we would like to perform


	//Explicit copy to device
	thrust::copy( hostVector.begin(), hostVector.end(), deviceVector.begin() );

	//Compute on device, here inclusive scan, for histogram equalization for instance
	thrust::inclusive_scan( deviceVector.begin(), deviceVector.end(), deviceVector.begin() );

	//Copy back to host
	thrust::copy( deviceVector.begin(), deviceVector.end(), hostVector.begin() );

	cudaStreamDestroy( stream0 );
	cudaStreamDestroy( stream1 );

	return EXIT_SUCCESS;
}
