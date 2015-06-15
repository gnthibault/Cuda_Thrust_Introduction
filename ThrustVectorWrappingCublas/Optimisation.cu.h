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
	//On of the solver parameter
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

	/***************************************
	 * Solving AX=B using gradient descent *
	 **************************************/

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

	//OpenMP backend performed 1000 iterations of gradient descent elements in 34.4254 seconds (0.0344254 seconds per iterations )
	//TBB backend performed 1000 iterations of gradient descent elements in 4.8519 seconds (0.0048519 seconds per iterations )
	//Cuda backend performed 1000 iterations of gradient descent elements in 0.731565 seconds (0.000731565 seconds per iterations )

	//CPU code Linked with MKL from Intel
	//TBB backend performed 1000 iterations of gradient descent elements in 4.87075 seconds (205.307 iterations per seconds )


};

void testVariationalSignalDenoising()
{

};


#endif /* OPTIMISATION_CU_H_ */
