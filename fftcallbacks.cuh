#ifndef FFTCALLBACKS_H
#define FFTCALLBACKS_H
#include <cuda.h>
#include <cuda_runtime.h>
#include <complex>
#include "device_basic_def.cuh"
#include "host_basic_def.hpp"
#include <cuComplex.h>
namespace cudaEM
{
	template<class T>
	__device__ T_complex<T> CB_ComplexValLD_C(
		void *dataIn, size_t offset,
		void *callerInfo, void *sharedPointer)
	{
		T_complex<T> prop_coeff = ((T_complex<T>*)callerInfo)[offset];
		T_complex<T> wave_val = ((T_complex<T>*)dataIn)[offset];
		T_complex<T> return_val;
		return_val.x = prop_coeff.x*wave_val.x - prop_coeff.y*wave_val.y;
		return_val.y = prop_coeff.x*wave_val.y + prop_coeff.y*wave_val.x;
		return return_val;
	}
	template<class T>
	__device__ T_complex<T> CB_ComplexValLD_CB(
		void *dataIn, size_t offset,
		void *callerInfo, void *sharedPointer)
	{
		size_t pixel_nm = PROBE_DIM*PROBE_DIM;
		T_complex<T> prop_coeff = ((T_complex<T>*)callerInfo)[offset%pixel_nm];
		T_complex<T> wave_val = ((T_complex<T>*)dataIn)[offset];
		T_complex<T> return_val;
		return_val.x = prop_coeff.x*wave_val.x - prop_coeff.y*wave_val.y;
		return_val.y = prop_coeff.x*wave_val.y + prop_coeff.y*wave_val.x;
		return return_val;
	}
	template<class T>
	__device__ T_complex<T> CB_ComplexValLD_R(
		void *dataIn, size_t offset,
		void *callerInfo, void *sharedPointer)
	{
		T prop_coeff = ((T*)callerInfo)[offset];
		T_complex<T> wave_val = ((T_complex<T>*)dataIn)[offset];
		T_complex<T> return_val, tempr;
		tempr.x = cos(prop_coeff);
		tempr.y = sin(prop_coeff);
		return_val.x = tempr.x*wave_val.x - tempr.y*wave_val.y;
		return_val.y = tempr.x*wave_val.y + tempr.y*wave_val.x;
		return return_val;
	}
	template<class T>
	__device__ T_complex<T> CB_ComplexValLD_RB(
		void *dataIn, size_t offset,
		void *callerInfo, void *sharedPointer)
	{
		size_t pixel_nm = PROBE_DIM*PROBE_DIM;
		T prop_coeff = ((T*)callerInfo)[offset%pixel_nm];
		T_complex<T> wave_val = ((T_complex<T>*)dataIn)[offset];
		T_complex<T> return_val, tempr;
		tempr.x = cos(prop_coeff);
		tempr.y = sin(prop_coeff);
		return_val.x = tempr.x*wave_val.x - tempr.y*wave_val.y;
		return_val.y = tempr.x*wave_val.y + tempr.y*wave_val.x;
		return return_val;
	}
	__device__ cufftCallbackLoadC d_loadCBCPtr_C = CB_ComplexValLD_C<float>;
	__device__ cufftCallbackLoadC d_loadCBCPtr_CB = CB_ComplexValLD_CB<float>;
	__device__ cufftCallbackLoadC d_loadCBCPtr_R = CB_ComplexValLD_R<float>;
	__device__ cufftCallbackLoadC d_loadCBCPtr_RB = CB_ComplexValLD_RB<float>;

	__device__ cufftCallbackLoadZ d_loadCBZPtr_C = CB_ComplexValLD_C<double>;
	__device__ cufftCallbackLoadZ d_loadCBZPtr_CB = CB_ComplexValLD_C<double>;
	__device__ cufftCallbackLoadZ d_loadCBZPtr_R = CB_ComplexValLD_C<double>;
	__device__ cufftCallbackLoadZ d_loadCBZPtr_RB = CB_ComplexValLD_C<double>;
}
#endif