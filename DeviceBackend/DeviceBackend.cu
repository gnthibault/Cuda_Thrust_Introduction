/*
 * DeviceBackend.cu
 *
 *  Created on: 12 juin 2015
 *      Author: Thibault Notargiacomo
 */


//STL
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <chrono>

//Thrust
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>

//Local
#include "DeviceBackend.cu.h"

int main( int argc, char* argv[] )
{
	const size_t sizeVect = 1024*1024*128;

	//Declare and allocate memory for device vector
	thrust::device_vector<float> deviceVector(sizeVect);

	//Counter used for random number generation
	thrust::counting_iterator<size_t> index_sequence_begin(0);

	//Generation of the random list of float
	thrust::transform(	index_sequence_begin,
						index_sequence_begin + sizeVect,
						deviceVector.begin(),
						normalRandomFunctor(0.f,10.f) );

	//Now measure how many time it take to perform sorting operation
	auto begin = std::chrono::high_resolution_clock::now();

	thrust::sort( deviceVector.begin(), deviceVector.end() );

	#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
		cudaDeviceSynchronize(); //Synchronize because of aynchronous behaviour in cuda mode
	#endif // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float> _elapsed = end - begin;
	double elapsed = _elapsed.count();
	double throughput = (sizeVect/1.e6)/elapsed;

	#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
		std::cout << "Cuda backend sorted " ;
	#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_OMP
		std::cout << "OpenMP backend sorted " ;
	#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_TBB
		std::cout << "TBB backend sorted " ;
	#endif // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA

	std::cout << sizeVect << " elements in " << elapsed << " seconds ("<< throughput <<" Millions of elements/s )"<< std::endl;

	return EXIT_SUCCESS;
}

//Cuda backend sorted 134217728 elements in 0.485675 seconds (276.353 Millions of elements/s )
//OpenMP backend sorted 134217728 elements in 224.313 seconds (0.598351 Millions of elements/s )
//TBB backend sorted 134217728 elements in 126.267 seconds (1.06297 Millions of elements/s )

