#pragma once
#include <Windows.h>
#include <vector>
#include <cstddef>
#include <thread>
#include <vector>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include "host_basic_def.hpp"
namespace cudaEM
{
	inline bool is_gpu_available()
	{
		bool is_available = false;
		try
		{
			int device_count = 0;
			cudaError_t error_id = cudaGetDeviceCount(&device_count);

			is_available = !((error_id != cudaSuccess) || (device_count == 0));
		}
		catch (...)
		{
			is_available = false;
		}

		return is_available;
	}

	inline int number_of_gpu_available()
	{
		int device_count = 0;
		cudaError_t error_id = cudaGetDeviceCount(&device_count);

		if (error_id != cudaSuccess)
		{
			device_count = 0;
		}
		return (device_count > 0) ? device_count : 0;
	}

	inline void host_memory_info(double &total, double &free)
	{
		MEMORYSTATUSEX status;
		status.dwLength = sizeof(status);
		GlobalMemoryStatusEx(&status);
		free = static_cast<double>(status.ullAvailPhys) / cSizeofMB;
		total = static_cast<double>(status.ullTotalPhys) / cSizeofMB;
	}
	inline void device_memory_info(double &total, double &free)
	{
		free = total = 0;
		std::size_t free_t, total_t;
		if (cudaSuccess == cudaMemGetInfo(&free_t, &total_t))
		{
			free = static_cast<double>(free_t) / cSizeofMB ;
			total = static_cast<double>(total_t) / cSizeofMB ;
		}
	}
	inline void get_device_properties(std::vector<Device_Properties> &device_properties)
	{
		device_properties.clear();

		if (!is_gpu_available())
		{
			return;
		}

		int device_count = 0;
		cudaGetDeviceCount(&device_count);
		device_properties.resize(device_count);
		for (auto idev = 0; idev < device_count; idev++)
		{
			cudaSetDevice(idev);
			cudaDeviceProp cuda_device_prop;
			cudaGetDeviceProperties(&cuda_device_prop, idev);

			device_properties[idev].id = idev;
			device_properties[idev].name = cuda_device_prop.name;
			device_properties[idev].compute_capability = 10 * cuda_device_prop.major + cuda_device_prop.minor;
			device_memory_info(device_properties[idev].total_memory_size, device_properties[idev].free_memory_size);
		}

		auto compare_fn = [](const Device_Properties &a, const Device_Properties &b)->bool { return a.compute_capability > b.compute_capability; };
		std::sort(device_properties.begin(), device_properties.end(), compare_fn);
	}
	inline void get_host_properties(Host_Properties &host_properties)
	{
		SYSTEM_INFO siSysInfo;
		GetSystemInfo(&siSysInfo);
		host_properties.nprocessors = siSysInfo.dwNumberOfProcessors;
		host_properties.nthreads = std::thread::hardware_concurrency();
		host_memory_info(host_properties.total_memory_size, host_properties.free_memory_size);
	}
}