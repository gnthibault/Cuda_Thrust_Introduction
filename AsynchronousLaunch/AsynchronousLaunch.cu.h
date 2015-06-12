/*
 * AsynchronousLaunch.cu.h
 *
 *  Created on: 12 juin 2015
 *      Author: Thibault Notargiacomo
 */

#ifndef ASYNCHRONOUSLAUNCH_CU_H_
#define ASYNCHRONOUSLAUNCH_CU_H_

/*
 * #include <thrust/for_each.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <cstdio>
struct printf_functor
{
  __host__ __device__
  void operator()(int x)
  {
    // note that using printf in a __device__ function requires
    // code compiled for a GPU with compute capability 2.0 or
    // higher (nvcc --arch=sm_20)
    printf("%d\n");
  }
};
...
thrust::device_vector<int> d_vec(3);
d_vec[0] = 0; d_vec[1] = 1; d_vec[2] = 2;
thrust::for_each_n(thrust::device, d_vec.begin(), d_vec.size(), printf_functor());
 *
 */
#endif /* ASYNCHRONOUSLAUNCH_CU_H_ */
