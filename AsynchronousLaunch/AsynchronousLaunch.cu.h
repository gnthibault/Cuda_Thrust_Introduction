/*
 * AsynchronousLaunch.cu.h
 *
 *  Created on: 12 juin 2015
 *      Author: Thibault Notargiacomo
 */

#ifndef ASYNCHRONOUSLAUNCH_CU_H_
#define ASYNCHRONOUSLAUNCH_CU_H_


//STL
#include <random>

//Create a functor that issues number following a normal distribution of specific mean and standard deviation
//Here we show how to write the same generator used in DeviceBackend.cu.h, but in full plain C++11 instead of thrust
template<typename T>
struct normalRandomFunctor
{
    normalRandomFunctor(T mean=0.f, T stddev=1.f) : m_generator(std::random_device()()), m_dist(mean, stddev) {};

	T operator()()
	{
		return m_dist(m_generator);
	}
private:
	//Mersenne Twister 19937 generator
	std::mt19937 m_generator;
	std::normal_distribution<T> m_dist;
};

template<typename T>
struct computeFunctor
{
	__host__ __device__
	computeFunctor() {}

	__host__ __device__
	T operator()( T in )
	{
		//Naive functor that generates expensive but useless instructions
		return pow(cos(in),2)+pow(sin(in),2);
	}
};
#endif /* ASYNCHRONOUSLAUNCH_CU_H_ */
