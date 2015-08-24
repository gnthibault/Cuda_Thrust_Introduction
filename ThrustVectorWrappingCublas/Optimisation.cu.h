/*
 * Optimisation.cu.h
 *
 *  Created on: Jun 15, 2015
 *      Author: Thibault Notargiacomo
 */

#ifndef OPTIMISATION_CU_H_
#define OPTIMISATION_CU_H_

//STL
#include <ctime>
#include <chrono>

//Local
#include "ThrustWrapper.cu.h"
#include "CublasWrapper.cu.h"


void testCublasWrapper()
{
	//Settings of the solver
	const size_t nbIteration = 1000;
	const double convergenceTol = 1e-12;

	//Declaring known data of the problem using random utilities
	const size_t sizeImage = 2000;
	const size_t sizeDomain = 3000;

	//Generates a matrix problem of size sizeImage*sizeDomain elements, with sizeDomain unknowns and sizeImage data
	CublasWrapper<float> A(sizeDomain,sizeImage);

	//Generates random Y vector
	ThrustVectorWrapper<float> B( sizeImage );
	B.FillWitNormalRandomValues();

	//Generates void solution vector
	ThrustVectorWrapper<float> X( sizeDomain );

	/****************************************************
	 * Solving min 0,5*||AX-B||² using gradient descent *
	 ****************************************************/

	//Declaring operand needed for
	ThrustVectorWrapper<float> Ax( sizeImage );
	ThrustVectorWrapper<float> Ag( sizeImage );
	ThrustVectorWrapper<float> grad( sizeDomain );

	/******************
	 * Main Algorithm *
	 ******************/

	//Initialization
	int niter = 0;
	double gradstep = 0;
	double L2Error = convergenceTol + 1;

	//Now measure how many time it takes to perform all iterations
	auto begin = std::chrono::high_resolution_clock::now();

	while( (niter < nbIteration) && (L2Error > convergenceTol) )
	{
		A.Prod( X, Ax );								// Ax = A * x
		Ax.Substract( B );								// Ax = Ax - b
		A.transProd( Ax, grad );						// grad = A^t(Ax - B)
		A.Prod( grad, Ag );								// Ag = A * gradient
		gradstep = grad.GetNorm22()/Ag.GetNorm22();		// Compute gradient step
		X.Saxpy( grad, -gradstep, false );				// Update solution

		L2Error = Ax.GetNorm22();						// Compute functional at current step
		std::cout <<"Iteration : "<<niter<< " over " <<nbIteration<<" , L2 error = " << L2Error << std::endl;

		niter++; 										// Ready for next iteration
	}

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float> _elapsed = end - begin;
	double elapsed = _elapsed.count();

	#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
		std::cout << "Cuda backend performed " ;
	#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_OMP
		std::cout << "OpenMP backend performed " ;
	#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_TBB
		std::cout << "TBB backend performed " ;
	#endif // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA

	std::cout << niter <<" iterations of gradient descent elements in " << elapsed << " seconds ("<< niter/elapsed <<" iterations per seconds )"<< std::endl;

	//CPU code linked with default gsl_cblas lib and default gcc gomp threading library
	//OpenMP backend performed 1000 iterations of gradient descent elements in 19.6776 seconds (50.8192 iterations per seconds )
	//TBB backend performed 1000 iterations of gradient descent elements in 13.6715 seconds (73.145 iterations per seconds )

	//CPU code Linked with MKL from Intel, and openMP runtime from intel (iomp5 instead of gomp
	//OpenMP backend performed 1000 iterations of gradient descent elements in 2.46626 seconds (405.473 iterations per seconds )
	//TBB backend performed 1000 iterations of gradient descent elements in 2.163 seconds (462.32 iterations per seconds )

	//Cuda Backend
	//Cuda backend performed 1000 iterations of gradient descent elements in 0.725926 seconds (1377.55 iterations per seconds )
};

void testVariationalSignalDenoising()
{
	std::cout << "***********************************************************************" << std::endl;
	std::cout << "****			Now benchmarking signal processing usecase			 ****" << std::endl;
	std::cout << "***********************************************************************" << std::endl;

	//Settings of the solver
	const size_t nbIteration = 10000;
	const double epsilonNorm = 1e-3;
	const double lambda = 1.1/5.0;
	const double stepSize = 4.0*( 1.8/(1.0+8.0*lambda/epsilonNorm) );


	//Declaring known data of the problem: a simple square wave for instance
	const size_t sizeSignal = 1024;
	const size_t halfOscillationPeriod = sizeSignal/16;
	ThrustVectorWrapper<float> Y( sizeSignal );

	//Simple loop that alternatively fill vector with 0's and 1's
	float value = 0;
	for( int i = 0; i < sizeSignal; i += halfOscillationPeriod )
	{
		thrust::fill( Y.GetDeviceVector().begin()+i, Y.GetDeviceVector().begin()+i+halfOscillationPeriod, value);
		value = (value==0) ? 1 : 0;
	}

	//Write perfect vector to CSV
	Y.SendToCSV();

	//Generate a gaussian noise of variance sigma
	const double sigma = 0.05;
	ThrustVectorWrapper<float> noise( sizeSignal );
	noise.FillWitGaussianRandomValues(0,sigma);

	//Add noise to original signal
	Y.Add( noise );

	//Write noisy vector to CSV
	Y.SendToCSV();

	//Unknown: the denoised version of the signal: initialized to 0
	ThrustVectorWrapper<float> X( sizeSignal );

	/***********************************************************************
	 * Solving min 0,5*||Y-X||² + \lambda Grad_e(x) using gradient descent *
	 ***********************************************************************/

	//Declaring operand needed for gradient descent
	ThrustVectorWrapper<float> TvGradientTmp( sizeSignal ); //Temporary variable used for calculation
	ThrustVectorWrapper<float> TvGradient( sizeSignal );
	ThrustVectorWrapper<float> grad( sizeSignal );

	/******************
	 * Main Algorithm *
	 ******************/

	//Initialization
	int niter = 0;

	//Now measure how many time it takes to perform all iterations
	auto begin = std::chrono::high_resolution_clock::now();

	while( niter < nbIteration )
	{
		grad.Assign( X );									// grad = X
		grad.Substract( Y );								// grad = X - Y
		TvGradientTmp.FiniteForwardDifference( X );			// TvGradient = G(X)
		TvGradientTmp.ApplySmoothedTVGradient(epsilonNorm);	// TvGradient = TvGradient / ||TvGradient||e
		TvGradient.FiniteBackwarDifference(TvGradientTmp);	// TvGradient = -div( TvGradient / ||TvGradient||e )
		grad.Saxpy( TvGradient, +lambda, false );			// grad = X - Y + lambda * GradientTV
		X.Saxpy( grad, -stepSize, false );					// Update solution

		std::cout <<"Iteration : "<<niter<< " over " << nbIteration << std::endl;

		niter++; 										// Ready for next iteration
	}

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float> _elapsed = end - begin;
	double elapsed = _elapsed.count();

	//Write denoised vector to CSV
	X.SendToCSV();

	#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
		std::cout << "Cuda backend performed " ;
	#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_OMP
		std::cout << "OpenMP backend performed " ;
	#elif THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_TBB
		std::cout << "TBB backend performed " ;
	#endif // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA

	std::cout << niter <<" iterations of gradient descent over "<<sizeSignal<<" elements in " << elapsed << " seconds ("<< niter/elapsed <<" iterations per seconds )"<< std::endl;

	//CPU code linked with default gcc gomp threading library
	//OpenMP backend performed 10000 iterations of gradient descent over 33554432 elements in 1672.89 seconds (5.97768 iterations per seconds )
	//TBB backend performed 10000 iterations of gradient descent over 33554432 elements in 1648.48 seconds (6.0662 iterations per seconds )

	//CPU code Linked with openMP runtime from intel (iomp5 instead of gomp )
	//OpenMP backend performed 10000 iterations of gradient descent over 33554432 elements in 1618.75 seconds (6.17761 iterations per seconds )

	//Cuda Backend
	//Cuda backend performed 10000 iterations of gradient descent over 33554432 elements in 105.78 seconds (94.5358 iterations per seconds )

};


#endif /* OPTIMISATION_CU_H_ */
