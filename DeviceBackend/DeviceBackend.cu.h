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
struct normalRandomFunctor
{
    __host__ __device__
    normalRandomFunctor(float a=0.f, float b=1.f) : m_mean(a), m_stddev(b) {};

    __host__ __device__
	float operator()(const size_t n) const
	{
		thrust::default_random_engine rng;
		thrust::random::normal_distribution<float> dist(m_mean, m_stddev);
		rng.discard(n);

		return dist(rng);
	}
private:
    float m_mean;
    float m_stddev;
};


#endif /* DEVICEBACKEND_CU_H_ */
