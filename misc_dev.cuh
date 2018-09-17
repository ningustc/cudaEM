#include <cuda.h>
#include <cuda_runtime.h>
namespace cudaEM
{
	template <class T>
	__device__ __forceinline__
		bool isEqual_dev(const T &a, const T &b);
	template <>
	__device__ __forceinline__
		bool isEqual_dev<int>(const int &a, const int &b)
	{
		return a == b;
	}
	template <>
	__device__ __forceinline__
		bool isEqual_dev<float>(const float &a, const float &b)
	{
		const float eps_abs = 1e-5f;
		const float eps_rel = 1e-4f;
		float diff = fabs(a - b);
		if (diff <= eps_abs)
			return true;
		return diff <= ((fabs(a) < fabs(b) ? fabs(b) : fabs(a))*eps_rel);
	}
	template <>
	__device__ __forceinline__
		bool isEqual_dev<double>(const double &a, const double &b)
	{
		const double eps_abs = 1e-13;
		const double eps_rel = 1e-8;
		double diff = fabs(a - b);
		if (diff <= eps_abs)
			return true;
		return diff <= ((fabs(a) < fabs(b) ? fabs(b) : fabs(a))*eps_rel);
	}
	template <class T>
	__device__ __forceinline__
		bool isZero_dev(const T &x)
	{
		return isEqual_dev<T>(x, 0);
	}
	template <class T, class U>
	__device__ __forceinline__
		bool isZero_dev(const T &x, const U &y)
	{
		return isEqual_dev<T>(x, 0) && isEqual_dev<U>(y, 0);
	}
	template <class T>
	__device__ __forceinline__
		bool nonZero_dev(const T &x)
	{
		return !isEqual_dev<T>(x, 0);
	}
	template <class T, class U>
	__device__ __forceinline__
		bool nonZero_dev(const T &x, const U &y)
	{
		return !(isEqual_dev<T>(x, 0) || isEqual_dev<U>(y, 0));
	}
}