/*
 * HostDeviceVector.cu.h
 *
 *  Created on: 12 juin 2015
 *      Author: Thibault Notargiacomo
 */

#ifndef HOSTDEVICEVECTOR_CU_H_
#define HOSTDEVICEVECTOR_CU_H_


//STL
#include <iostream>

//Thrust
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/system_error.h>

//Cuda
#include <cuda_runtime.h>

//Local
#include "../Include/cudaHelper.cu.h"

#define VEC_SIZE 8

//Version 1: Thrust as a nice wrapper for generic algorithms
void version1()
{
	//Thrust Device vectors intend to mimic std::vector class from stl, plus its algorithms
	thrust::device_vector<int> deviceVector;
	//Also available in host flavour
	thrust::host_vector<int> hostVector;

	//Allocate vector on device
	deviceVector.resize( VEC_SIZE );
	//Initialize host vector as size 8 elements, each containing the value 111
	hostVector.resize( VEC_SIZE, 111 );

	//Explicit copy to device
	thrust::copy( hostVector.begin(), hostVector.end(), deviceVector.begin() );

	//Compute on device, here inclusive scan, for histogram equalization for instance
	thrust::inclusive_scan( deviceVector.begin(), deviceVector.end(), deviceVector.begin() );

	//Copy back to host
	thrust::copy( deviceVector.begin(), deviceVector.end(), hostVector.begin() );

	//Print results
	std::cout << "Version 1, vector contains: ";
	for( auto it = hostVector.begin(); it != hostVector.end(); it++ )
	{
		std::cout << " / " << *it;
	}
	std::cout << std::endl;
}

//Version 2: Thrust for fast prototyping/debugging
//Here it is interesting to notice that we are dereferencing an iterator to device vector !
void version2()
{
	//Declare and initialize device vector in one line
	thrust::device_vector<int> deviceVector( VEC_SIZE, 111 );

	//Compute algorithm
	thrust::inclusive_scan( deviceVector.begin(), deviceVector.end(), deviceVector.begin() );

	//Print results
	std::cout << "Version 2, vector contains: ";
	for( auto it = deviceVector.begin(); it != deviceVector.end(); it++ )
	{
		std::cout << " / " << *it;  //Dereferencing iterator for reading: can also be done for writing !
	}
	std::cout << std::endl;
}

//Version3 : thrust algorithm can also handle user allocated memory
void version3()
{
	//Raw pointer to device memory
	int * raw_ptr;
	checkCudaErrors( cudaMalloc((void **) &raw_ptr, VEC_SIZE * sizeof(int) ) );

	//Wrap raw pointer with a device_ptr
	thrust::device_ptr<int> dev_ptr(raw_ptr);

	//Use device_ptr in thrust algorithms
	thrust::fill(dev_ptr, dev_ptr + VEC_SIZE, (int) 111);

	//Compute on device, here inclusive scan, for histogram equalization for instance
	thrust::inclusive_scan( dev_ptr, dev_ptr + VEC_SIZE, dev_ptr );

	//Print results
	std::cout << "Version 3, vector contains: ";
	for( int i = 0; i != VEC_SIZE; i++ )
	{
		std::cout << " / " << dev_ptr[i]; //Dereferencing pointer for reading: can also be done for writing !
	}
	std::cout << std::endl;

	// free memory
	checkCudaErrors( cudaFree(raw_ptr) );
}

template<typename T, size_t SIZE>
__global__ void naive_sequential_scan( T* ptr )
{
	T val = 0;
	#pragma unroll
	for( auto i = 0; i < SIZE; i++ )
	{
		ptr[i] += val;
		val = ptr[i];
	}
}

//Version4 : User can also handle Thrust allocated memory
void version4()
{
	//Declare and initialize device vector in one line
	thrust::device_vector<int> deviceVector( VEC_SIZE, 111 );

	//Compute algorithm
	cudaStream_t stream;
	checkCudaErrors( cudaStreamCreate(&stream) );
	naive_sequential_scan<int,VEC_SIZE><<<1,1,0,stream>>>( thrust::raw_pointer_cast(deviceVector.data() ) );
	checkCudaErrors( cudaStreamSynchronize( stream) );

	//Print results
	std::cout << "Version 4, vector contains: ";
	for( auto it = deviceVector.begin(); it != deviceVector.end(); it++ )
	{
		std::cout << " / " << *it;  //Dereferencing iterator for reading: can also be done for writing !
	}
	std::cout << std::endl;
	checkCudaErrors( cudaStreamDestroy(stream) );
}

//This version only intend to show that thrust allows to handle error
//through a proper exception handling mechanism
void version5()
{
	try
	{
		//Declare and initialize device vector in one line
		thrust::device_vector<int> deviceVector( VEC_SIZE, 111 );

		//Compute algorithm
		std::cout << "Version 5, we are going to catch an exception: ";
		thrust::inclusive_scan( deviceVector.begin(), deviceVector.end()+1, deviceVector.begin() ); //This line purposely contains an error

		//Print results
		for( auto it = deviceVector.begin(); it != deviceVector.end(); it++ )
		{
			std::cout << " / " << *it;
		}
	} catch( thrust::system_error &e )
	{
		std::cerr << "Thrust mechanism for handling error in version5() caused by : " << e.what() << std::endl;
	}
}

#endif /* HOSTDEVICEVECTOR_CU_H_ */
