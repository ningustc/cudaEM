#ifndef DEVICE_BASIC_DEF_H
#define DEVICE_BASIC_DEF_H
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>
#define C_PI_DEV 3.141592653589793238463
#define C_2PI_DEV 6.283185307179586476925
namespace cudaEM
{
	/***********************definitions for the basic device types***********************/
	template <class T>
	using T_complex = typename std::conditional<std::is_same<T, float>::value, cuComplex, cuDoubleComplex>::type;
	template <class T>
	using T_C = thrust::complex<T>;
	template <class T>
	using TVector_Dev = thrust::device_vector<T>;
	template <class T>
	using TVector_Dev_C = thrust::device_vector<T_complex<T>>;
}
#endif