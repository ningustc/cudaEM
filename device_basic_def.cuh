#ifndef DEVICE_BASIC_DEF_H
#define DEVICE_BASIC_DEF_H
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>
#include <cufftXt.h>
#define C_PI_DEV 3.141592653589793238463
#define C_2PI_DEV 6.283185307179586476925
namespace cudaEM
{
	/***********************definitions for the basic device types***********************/
	template <class T>
	using T_complex = typename std::conditional<std::is_same<T, float>::value, cuComplex, cuDoubleComplex>::type;
	template <class T>
	using T_Callback = typename std::conditional<std::is_same<T, float>::value, cufftCallbackLoadC, cufftCallbackLoadZ>::type;
	template <class T>
	cufftXtCallbackType_t get_callback_type()
	{
		if (std::is_same<T, float>::value)
		{
			return CUFFT_CB_LD_COMPLEX;
		}
		else
		{
			return CUFFT_CB_LD_COMPLEX_DOUBLE;
		}
	}
}
#endif