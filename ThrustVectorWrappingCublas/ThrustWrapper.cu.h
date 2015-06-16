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
#include <fstream>

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
#include <thrust/adjacent_difference.h>

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
	//Output to end of .csv
	void SendToCSV( std::string strName = std::string( "Denoising.csv") )
	{
		std::ofstream CSVToFile( strName, std::ios_base::app );
		for( auto it = m_deviceVector.begin(); it != m_deviceVector.end(); it++)
		{
			if (it != m_deviceVector.begin())
			{
					CSVToFile << ",";
			}
			CSVToFile << *it;
		}
		CSVToFile << std::endl;
	}

	//Vector operators
	void Assign( const ThrustVectorWrapper<T>& Input )
	{
		try
		{
			thrust::copy( Input.GetConstDeviceVector().begin(), Input.GetConstDeviceVector().end(),	m_deviceVector.begin() );
		}
		catch( thrust::system_error &e )
		{
			std::cerr << "ThrustVectorWrapper::Assign("<< &Input <<") error: " << e.what() << std::endl;
			exit(-1);
		}
	}
	void Add( const ThrustVectorWrapper<T>& Input )
	{
		const thrust::device_vector<T>& in = Input.GetConstDeviceVector();
		try
		{
			thrust::transform( m_deviceVector.begin(), m_deviceVector.end(), in.begin(), m_deviceVector.begin(), thrust::plus<T>() );
		}
		catch( thrust::system_error &e )
		{
			std::cerr << "ThrustVectorWrapper::Add("<<&Input<<") error: " << e.what() << std::endl;
			exit(-1);
		}
	}
	void Substract( ThrustVectorWrapper<T>& Input )
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
	void Saxpy( ThrustVectorWrapper<T>& Input, const double scalarInput, bool isMemberMultipliedByScalar )
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
	// FiniteDifference : compute Input[0]=m_dev[0] and Input[n] = m_dev[n]-m_dev[n-1]
	void FiniteForwardDifference( const ThrustVectorWrapper<T>& Input )
	{
		const thrust::device_vector<T>& in = Input.GetConstDeviceVector();
		try
		{
			thrust::adjacent_difference( in.begin(), in.end(), m_deviceVector.begin());
		}
		catch( thrust::system_error &e )
		{
			std::cerr << "ThrustVectorWrapper::FiniteForwardDifference("<<&Input<<") error: " << e.what() << std::endl;
			exit(-1);
		}
	}
	// FiniteDifference : compute Input[last]=m_dev[last] and Input[n] = m_dev[n+1]-m_dev[n]
	void FiniteBackwarDifference( const ThrustVectorWrapper<T>& Input )
	{
		const thrust::device_vector<T>& in = Input.GetConstDeviceVector();
		try
		{
			thrust::transform( in.begin()+1, in.end(), in.begin(), m_deviceVector.begin(), thrust::minus<T>() );
			*(m_deviceVector.end()-1) = *(in.end()-1);
		}
		catch( thrust::system_error &e )
		{
			std::cerr << "ThrustVectorWrapper::FiniteBackwarDifference("<<&Input<<") error: " << e.what() << std::endl;
			exit(-1);
		}
	}

	// ApplySmoothedTVGradient
	void ApplySmoothedTVGradient( T epsilon )
	{
		try
		{
			thrust::transform( m_deviceVector.begin(), m_deviceVector.end(), m_deviceVector.begin(), smoothedTVGradient_functor(epsilon) );
		}
		catch( thrust::system_error &e )
		{
			std::cerr << "ThrustVectorWrapper::ApplySmoothedTVGradient("<<epsilon<<") error: " << e.what() << std::endl;
			exit(-1);
		}
	}

	//Norm-style operator
	double GetNorm22() const
	{
		double result = 1.0;
		try
		{
			result = thrust::inner_product( m_deviceVector.begin(), m_deviceVector.end(), m_deviceVector.begin(), 0.0 );
		}
		catch( thrust::system_error &e )
		{
			std::cerr << "ThrustVectorWrapper::GetNorm22() error: " << e.what() << std::endl;
			exit(-1);
		}
		return result;
	}
	void FillWitGaussianRandomValues( T mean = 0, T stddev = 1)
	{
		GaussianRandomFunctor g(mean,stddev);
		_FillWitRandomFunctor( g );
	}
	void FillWitNormalRandomValues( T min = 0, T max = 1)
	{
		UniformRandomFunctor u(min,max);
		_FillWitRandomFunctor( u );
	}
protected:
	template< class Func >
	void _FillWitRandomFunctor( Func& functor )
	{
		try
		{
			srand( time(NULL)); //Random Seed
			size_t seed = rand()+m_sRandomSeed;
			thrust::transform( 	thrust::make_counting_iterator<size_t>( seed),
								thrust::make_counting_iterator<size_t>( seed+m_deviceVector.size() ),
								m_deviceVector.begin(),
								functor );
			m_sRandomSeed += m_deviceVector.size();
		}
		catch( thrust::system_error &e )
		{
			std::cerr << "ThrustVectorWrapper::FillWitRandomFunctor(), error: " << e.what() << std::endl;
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
	struct UniformRandomFunctor : public thrust::unary_function<size_t,T>
	{
		UniformRandomFunctor(T min=0, T max=1) : m_min(min), m_max(max) {};

		__host__ __device__
		T operator()( const size_t n) const
		{
			//see http://docs.thrust.googlecode.com/hg/group__random__number__engine__adaptors.html for discarding behaviour
			thrust::default_random_engine rng;
			thrust::uniform_real_distribution<T> uniform_distribution(m_min, m_max);
			rng.discard( n );
			return uniform_distribution( rng );
		};
		const T m_min;
		const T m_max;
	};
	struct GaussianRandomFunctor : public thrust::unary_function<size_t,T>
	{
		GaussianRandomFunctor(T a=0.f, T b=1.f) : m_mean(a), m_stddev(b) {};

		__host__ __device__
		T operator()(const size_t n) const
		{
			thrust::default_random_engine rng;
			thrust::random::normal_distribution<T> gaussian_distribution(m_mean, m_stddev);
			rng.discard(n);
			return gaussian_distribution(rng);
		}
	private:
		T m_mean;
		T m_stddev;
	};

	struct smoothedTVGradient_functor : public thrust::unary_function<T,T>
	{
		smoothedTVGradient_functor( T epsilon ) : m_epsilonSquared( pow(epsilon,2) ) {};

		__host__ __device__
		T operator()( T in ) const
		{
			return in / sqrt( m_epsilonSquared+in*in );
		};
	private:
		T m_epsilonSquared;
	};

protected:
	thrust::device_vector<T> m_deviceVector;
	static size_t m_sRandomSeed;
};
template<typename T> size_t ThrustVectorWrapper<T>::m_sRandomSeed = 0;


#endif /* THRUSTWRAPPER_CU_H_ */
