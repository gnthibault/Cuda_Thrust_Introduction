/*
 * ThrustWrapper.cu.h
 *
 *  Created on: Jun 15, 2015
 *      Author: Thibault Notargiacomo
 */

#ifndef THRUSTWRAPPER_CU_H_
#define THRUSTWRAPPER_CU_H_

// STL
#include <ctime>
#include <iostream>

//Thrust
#include <thrust/system_error.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/inner_product.h>
#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>


template<typename T>
class ThrustVectorWrapper
{
public:
	//Ctor
	ThrustVectorWrapper(){};
	ThrustVectorWrapper( size_t size, T initValue = 0 )
	{
		try
		{
			Resize( size, initValue );
		}
		catch( thrust::system_error &e )
		{
			std::cerr << "ThrustVectorWrapper::ThrustVectorWrapper("<< size <<") error: " << e.what() << std::endl;
			exit(-1);
		}
	};

	//Setters
	void Resize( size_t size, T initValue = 0 )
	{
		m_deviceVector.resize( size, initValue );
	}

	//Getters
	thrust::device_vector<T>& GetDeviceVector()
	{
		return m_deviceVector;
	}
	const thrust::device_vector<T>& GetConstDeviceVector() const
	{
		return m_deviceVector;
	}
	virtual void Substract( ThrustVectorWrapper<T>& Input )
	{
		const thrust::device_vector<T>& in = Input.GetConstDeviceVector();
		try
		{
			thrust::transform( m_deviceVector.begin(), m_deviceVector.end(), in.begin(), m_deviceVector.begin(), thrust::minus<T>() );
		}
		catch( thrust::system_error &e )
		{
			std::cerr << "ThrustVectorWrapper::Substract("<<&Input<<") error: " << e.what() << std::endl;
			exit(-1);
		}
	}
	//Saxpy: X <- AX + Y  or X <- AY + X , X being the member vector
	virtual void Saxpy( ThrustVectorWrapper<T>& Input, const double scalarInput, bool isMemberMultipliedByScalar )
	{
		const thrust::device_vector<T>& in = Input.GetConstDeviceVector();
		try
		{
			if( isMemberMultipliedByScalar ) // X <- AX + Y
			{
				thrust::transform( m_deviceVector.begin(), m_deviceVector.end(), in.begin(), m_deviceVector.begin(), saxpy_functor(scalarInput) );
			}else // X <- AY + X
			{
				thrust::transform( in.begin(), in.end(), m_deviceVector.begin(), m_deviceVector.begin(), saxpy_functor( scalarInput ) );
			}
		}
		catch( thrust::system_error &e )
		{
			std::cerr << "ThrustVectorWrapper::Saxpy("<< &Input <<") error: " << e.what() << std::endl;
			exit(-1);
		}
	}

	//Norm-style operator
	virtual double GetNorm0() const
	{
		double result = 0.0;
		try
		{
			result = thrust::reduce( m_deviceVector.begin(), m_deviceVector.end(), 0.0, norm0_functor() );
		}
		catch( thrust::system_error &e )
		{
			std::cerr << "ThrustVectorWrapper::GetNorm0() error: " << e.what() << std::endl;
		}
		return result;
	}
	virtual double GetNorm22() const
	{
		double result = 1.0;
		try
		{
			result = thrust::inner_product( m_deviceVector.begin(), m_deviceVector.end(), m_deviceVector.begin(), 0.0 );
		}
		catch( thrust::system_error &e )
		{
			std::cerr << "ThrustVectorWrapper::GetNorm22() error: " << e.what() << std::endl;
		}
		return result;
	}
	virtual void FillWitNormalRandomValues( double min = 0, double max = 1)
	{
		try
		{
			srand( time(NULL)); //Random Seed
			size_t seed = rand()+m_sRandomSeed;
			thrust::transform( 	thrust::make_counting_iterator<size_t>( seed),
								thrust::make_counting_iterator<size_t>( seed+m_deviceVector.size() ),
								m_deviceVector.begin(),
								UniformRandomFunctor(min,max));
			m_sRandomSeed += m_deviceVector.size();
		}
		catch( thrust::system_error &e )
		{
			std::cerr << "ThrustVectorWrapper::FillWitNormalRandomValues(" << min << " , " << max << "), error: " << e.what() << std::endl;
			exit(-1);
		}
	}

public:	//Specific functors for use with Thrust::transform
	struct saxpy_functor : public thrust::binary_function<T,T,T>
	{
		saxpy_functor(T a) : m_a(a) {};
		__host__ __device__
		T operator()(const T& x, const T& y) const
		{
			return m_a * x + y;
		};
		const T m_a;
	};
	struct norm0_functor : public thrust::binary_function<T,T,T>
	{
		norm0_functor() {};
		__host__ __device__
		T operator()(const T& x, const T& y) const
		{

			return y != 0 ? x+1 : x;
		};
	};
	struct UniformRandomFunctor : public thrust::binary_function<size_t,T,T>
	{
		UniformRandomFunctor(T min=0, T max=1) : m_min(min), m_max(max) {};

		__host__ __device__
		T operator()(size_t n) const
		{
			//see http://docs.thrust.googlecode.com/hg/group__random__number__engine__adaptors.html for discarding behaviour
			thrust::default_random_engine random_engine;
			thrust::uniform_real_distribution<T> uniform_distribution(m_min, m_max);
			random_engine.discard( n );
			return uniform_distribution( random_engine );
		};
		const T m_min;
		const T m_max;
	};

protected:
	thrust::device_vector<T> m_deviceVector;
	static size_t m_sRandomSeed;
};
template<typename T> size_t ThrustVectorWrapper<T>::m_sRandomSeed = 0;


#endif /* THRUSTWRAPPER_CU_H_ */
