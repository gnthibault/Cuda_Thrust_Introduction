/*
 * CublasWrapper.cu.h
 *
 *  Created on: Jun 15, 2015
 *      Author: Thibault Notargiacomo
 */

#ifndef CUBLASWRAPPER_CU_H_
#define CUBLASWRAPPER_CU_H_

// cublas
#include <cublas_v2.h>

#ifdef USE_GSL_CBLAS
	#include <gsl/gsl_blas.h>
#endif //USE_GSL_CBLAS

//Local
#include "ThrustWrapper.cu.h"

template<typename T>
struct cublasHelper
{
	cublasHelper(){};
	static void Xgemv(cublasHandle_t handle, cublasOperation_t transA,  const T *A, const T *B, T *C, const int m, const int n){};
	static void ControlCublasErrorStatus( cublasStatus_t cudaStat, std::string strFuncName = std::string("cublasHelper::UnknownFunction") )
	{
		if( cudaStat != CUBLAS_STATUS_SUCCESS )
		{
			std::cout << strFuncName << "Error encountered in cublasHelper" << std::endl;
		}
	};
};

//Specialisation for floating point operations
template<>
void cublasHelper<float>::Xgemv(cublasHandle_t handle, cublasOperation_t transA,  const float *A, const float *B, float *C, const int m, const int n)
{
	const float alf = 1;
	const float bet = 0;

	#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
		const float *alpha = &alf;
		const float *beta = &bet;
		int lda = m;
		int ldb = 1;
		int ldc = 1;
		// Do the actual multiplication
		ControlCublasErrorStatus( cublasSgemv(handle, transA, m, n, alpha, A, lda, B, ldb, beta, C, ldc), std::string("cublasHelper<float>::Xgemv") );

	#else //THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
		#if defined USE_GSL_CBLAS
			//Attention, cublas stocke les matrices en Column major order, comme fortran, alors que gsl les stocke en Row major order, comme en C classique
			//m1 is considered as the input vector size, n1 as the output
			int m1 = (transA == CUBLAS_OP_N) ? n : m;
			int n1 = (transA == CUBLAS_OP_N) ? m : n;
			//For A, we have m : nb of rows, n: nb of columns
			const gsl_matrix_float_const_view gA = gsl_matrix_float_const_view_array(A, n, m);
			const gsl_vector_float_const_view gB = gsl_vector_float_const_view_array(B, m1 );
			gsl_vector_float_view gC = gsl_vector_float_view_array( C, n1 );

			// Do the actual multiplication
			gsl_blas_sgemv( (transA != CUBLAS_OP_N) ? CblasNoTrans : CblasTrans, alf, &gA.matrix, &gB.vector, bet, &gC.vector);

		#else //defined USE_GSL_CBLAS
			#error "No GSL CBLAS installed, nor THRUST_DEVICE_SYSTEM_CUDA is defined, no linear algebra operation possible"
		#endif //defined USE_GSL_CBLAS
	#endif // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
};

//Specialisation for double precision floating point operations
template<>
void cublasHelper<double>::Xgemv(cublasHandle_t handle, cublasOperation_t transA,  const double *A, const double *B, double *C, const int m, const int n)
{
	const double alf = 1;
	const double bet = 0;

	#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
		const double *alpha = &alf;
		const double *beta = &bet;
		int lda = m;
		int ldb = 1;
		int ldc = 1;
		// Do the actual multiplication
		ControlCublasErrorStatus( cublasDgemv(handle, transA, m, n, alpha, A, lda, B, ldb, beta, C, ldc), std::string("cublasHelper<double>::Xgemv"));

	#else //THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
		#if defined USE_GSL_CBLAS
			//Attention, cublas stocke les matrices en Column major order, comme fortran, alors que gsl les stocke en Row major order, comme en C classique
			//m1 is considered as the input vector size, n1 as the output
			int m1 = (transA == CUBLAS_OP_N) ? n : m;
			int n1 = (transA == CUBLAS_OP_N) ? m : n;
			//For A, we have m : nb of rows, n: nb of columns
			const gsl_matrix_const_view gA = gsl_matrix_const_view_array(A, n, m);
			const gsl_vector_const_view gB = gsl_vector_const_view_array(B, m1 );
			gsl_vector_view gC = gsl_vector_view_array(C, n1 );
			// Do the actual multiplication
			gsl_blas_dgemv( (transA != CUBLAS_OP_N) ? CblasNoTrans : CblasTrans, alf, &gA.matrix, &gB.vector, bet, &gC.vector);

		#else //defined USE_GSL_CBLAS
			#error "No GSL CBLAS installed, nor THRUST_DEVICE_SYSTEM_CUDA is defined, no linear algebra operation possible"
		#endif //defined USE_GSL_CBLAS
	#endif // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
};

template<typename T>
class CublasWrapper
{
public :
	CublasWrapper( int sizeDomain, int sizeImage, T minVal=0, T maxVal=1 ): m_sizeDomain(sizeDomain), m_sizeImage(sizeImage)
	{
		//We generate a random linear problem
		m_deviceMatrix.Resize( sizeDomain*sizeImage );
		m_deviceMatrix.FillWitNormalRandomValues(minVal,maxVal);
		cublasCreate( &m_cuBlasHandle );
	};

	~CublasWrapper()
	{
		 // Destroy the handle
		 cublasDestroy( m_cuBlasHandle );
	};

	void Prod(const ThrustVectorWrapper<T>& Input, ThrustVectorWrapper<T>& Output) const
	{
		// Multiply the arrays A and B on GPU and save the result in C
		// C(m,n) = A(m,k) * B(k,n)
		cublasHelper<T>::Xgemv( m_cuBlasHandle,
					CUBLAS_OP_N,	// the non-transpose operation is selected
					thrust::raw_pointer_cast(m_deviceMatrix.GetConstDeviceVector().data()),
					thrust::raw_pointer_cast(Input.GetConstDeviceVector().data()),
					thrust::raw_pointer_cast(Output.GetDeviceVector().data()),
					m_sizeImage,
					m_sizeDomain );
	};

	void transProd(const ThrustVectorWrapper<T>& Input, ThrustVectorWrapper<T>& Output) const
	{
		cublasHelper<T>::Xgemv( m_cuBlasHandle,
					CUBLAS_OP_T,	// the transpose operation is selected
					thrust::raw_pointer_cast(m_deviceMatrix.GetConstDeviceVector().data()),
					thrust::raw_pointer_cast(Input.GetConstDeviceVector().data()),
					thrust::raw_pointer_cast(Output.GetDeviceVector().data()),
					m_sizeImage,
					m_sizeDomain );
	};

protected:
	int m_sizeDomain;
	int m_sizeImage;
	ThrustVectorWrapper<T> m_deviceMatrix;	//The actual linear system matrix
	cublasHandle_t m_cuBlasHandle;
};

#endif /* CUBLASWRAPPER_CU_H_ */
