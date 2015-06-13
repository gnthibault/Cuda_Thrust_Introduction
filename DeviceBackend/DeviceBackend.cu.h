/*
 * DeviceBackend.cu.h
 *
 *  Created on: 12 juin 2015
 *      Author: Thibault Notargiacomo
 */

#ifndef DEVICEBACKEND_CU_H_
#define DEVICEBACKEND_CU_H_


//Thrust
#include <thrust/random.h>

//Create a functor that issues number following a normal distribution of specific mean and standard deviation
template<typename T>
struct normalRandomFunctor
{
    __host__ __device__
    normalRandomFunctor(T a=0.f, T b=1.f) : m_mean(a), m_stddev(b) {};

    __host__ __device__
	float operator()(const size_t n) const
	{
		thrust::default_random_engine rng;
		thrust::random::normal_distribution<T> dist(m_mean, m_stddev);
		rng.discard(n);

		return dist(rng);
	}
private:
    T m_mean;
    T m_stddev;
};


#endif /* DEVICEBACKEND_CU_H_ */
