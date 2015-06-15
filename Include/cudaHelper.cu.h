inline void __checkCudaErrors( cudaError err, const char *file, const int line )
{
	if( err != cudaSuccess )
	{
		printf("%s(%i) : CUDA Runtime API error %i : %s \n",file ,line, (int)err, cudaGetErrorString(err) );
	}
};

#define checkCudaErrors(err)	__checkCudaErrors (err, __FILE__, __LINE__)
