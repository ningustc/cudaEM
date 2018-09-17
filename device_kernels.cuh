#ifndef DEVICE_KERNELS_H
#define DEVICE_KERNELS_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuComplex.h>
#include <device_functions.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>
#include "host_basic_def.hpp"
#include "device_basic_def.cuh"
#include "grid_dev.cuh"
#include "device_containers.cuh"
#include "basic_functions_dev.cuh"
namespace cudaEM
{
	inline double GPU_free_memory_info()
	{
		double free = 0;
		size_t free_t, total_t;
		if (cudaSuccess == cudaMemGetInfo(&free_t, &total_t))
		{
			free = static_cast<double>(free_t);
		}
		return free;
	};
	template <class T>
	__global__ void gen_slice_potential_kernel(Grid_2d_Dev<T> grid_2d, Combined_Element_Coeff_Ptrs<T>* atoms, T* V, int n_atoms, T coeff)
	{
		int Thr_idx = threadIdx.x + blockIdx.x*blockDim.x;
		int Thr_idy = threadIdx.y + blockIdx.y*blockDim.y;
		if (Thr_idx< grid_2d.col && Thr_idy <grid_2d.row)
		{
			for (auto iatom = 0; iatom < n_atoms; iatom++)
			{
				//calculate the distance between the atom and pixel.
				const auto R_squared = grid_2d.Distance_square(Thr_idx, Thr_idy, atoms[iatom].x, atoms[iatom].y);
				if (R_squared < atoms[iatom].R_max_squared)
				{
					const T Voltage = coeff*atoms[iatom].occ*get_value_Cubic_Spline(R_squared, atoms[iatom]);
					const size_t xy_idx = grid_2d.row_major_idx(Thr_idx, Thr_idy);
					V[xy_idx] += Voltage;
				}
			}
		}
	}
	template <class T>
	__global__ void gen_sa_coeff_kernel(ePotential_Type potential_type, T* qz,
		Combined_Element_Coeff_Ptrs<T>* atom, T z_interval)
	{
		/*******shared to be added*********/
		int x_idx = threadIdx.x;
		int R_idx = blockIdx.x;
		int atom_idx = blockIdx.y;
		/*****Attention!!! the size should be const*****/
		__shared__ T V0s[c_nqz];
		__shared__ T dV0s[c_nqz];

		//x coordinate and corresponding weight
		T x = qz[x_idx * 2];
		T w = z_interval*qz[x_idx * 2 + 1];
		T R2 = atom[atom_idx].R2_dev[R_idx];

		T V, dVir;
		T a = atom[atom_idx].z_middle;
		T z = z_interval*x + a;
		//the distance to the sphere center
		T r = sqrt(z*z + R2);
		Vr_dVrir_dev<T>(potential_type, r, atom[atom_idx].cl_dev, atom[atom_idx].cnl_dev, w, V, dVir);
		V0s[x_idx] = V;
		dV0s[x_idx] = dVir;
		__syncthreads();
		int i = c_nqz / 2;
		while (i != 0)
		{
			if (x_idx < i)
			{
				V0s[x_idx] += V0s[x_idx + i];
				dV0s[x_idx] += dV0s[x_idx + i];
			}
			__syncthreads();
			i /= 2;
		}
		if (x_idx == 0)
		{
			V = V0s[0];
			dVir = 0.5*dV0s[0];
			auto R2_tap = atom[atom_idx].R_tap_squared;
			auto tap_cf = atom[atom_idx].tap_cf;
			apply_tapering_dev(R2_tap, tap_cf, R2, V, dVir);
			atom[atom_idx].c0_dev[R_idx] = V;
			atom[atom_idx].c1_dev[R_idx] = dVir;
		}
	}
	template <class T>
	__global__ void fill_spl_coeff_kernel(Combined_Element_Coeff_Ptrs<T>* atom)
	{
		int R_idx = threadIdx.x;
		int atom_idx = blockIdx.x;
		if (R_idx < blockDim.x - 1)
		{
			get_cubic_poly_coef(R_idx, atom[atom_idx]);
		}
	}
	template <class T>
	__global__ void generate_trans_function(T w, T* Vp, T_complex<T>* Trans, size_t nxy)
	{
		size_t thr_idx = threadIdx.x+ blockIdx.x*blockDim.x;
		size_t thr_num = blockDim.x*gridDim.x;
		T theta;
		while (thr_idx< nxy)
		{
			theta = Vp[thr_idx]*w;
			Trans[thr_idx].x = cos(theta);
			Trans[thr_idx].y = sin(theta);
			thr_idx += thr_num;
		}
	}
	template <class T>
	__global__ void bandwidth_limit(Grid_2d_Dev<T> grid_2d, T_complex<T>* Wave)
	{
		int thr_idx = threadIdx.x + blockIdx.x*blockDim.x;
		int thr_idy = threadIdx.y + blockIdx.y*blockDim.y;

		if ((thr_idx < grid_2d.col) && (thr_idy < grid_2d.row))
		{
			const size_t ixy = grid_2d.row_major_idx(thr_idx, thr_idy);
			T bwl_factor = grid_2d.bwl_factor_shift(thr_idx, thr_idy) / grid_2d.Grid_size_float();
			Wave[ixy].x *= bwl_factor;
			Wave[ixy].y *= bwl_factor;
		}
	}
	template <class T>
	__global__ void modulate_wave_kernel(T w, T* Vp, 
		T_complex<T>* wave_io, size_t nxy)
	{
		size_t thr_idx = threadIdx.x + blockIdx.x*blockDim.x;
		size_t thr_num = blockDim.x*gridDim.x;
		T_complex<T> tempr;
		T theta;
		while (thr_idx < nxy)
		{
			theta = Vp[thr_idx] * w;
			tempr.x = cos(theta);
			tempr.y = sin(theta);
			wave_io[thr_idx] = complex_Mul<T>(wave_io[thr_idx], tempr);
			thr_idx += thr_num;
		}
	}
	template <class T>
	__global__ void modulate_waves_kernel(T w, T* Vp,
		T_complex<T>* wave_io,int batch_num , size_t nxy)
	{
		size_t thr_idx = threadIdx.x + blockIdx.x*blockDim.x;
		size_t thr_num = blockDim.x*gridDim.x;
		T_complex<T> tempr;
		T theta;
		while (thr_idx < nxy)
		{
			theta = Vp[thr_idx] * w;
			tempr.x = cos(theta);
			tempr.y = sin(theta);
			for (int i=0; i< batch_num; i++)
			{
				wave_io[thr_idx+i*nxy] = complex_Mul<T>(wave_io[thr_idx + i*nxy], tempr);
			}
			thr_idx += thr_num;
		}
	}
	template <class T>
	__global__ void set_wave_val_kernel(T_complex<T>* wave_io, T_complex<T> c_val, size_t nxy)
	{
		size_t thr_idx = threadIdx.x + blockIdx.x*blockDim.x;
		size_t thr_num = blockDim.x*gridDim.x;
		while (thr_idx < nxy)
		{
			wave_io[thr_idx]=c_val;
			thr_idx += thr_num;
		}
	}
	template <class T>
	__global__ void set_intensity_val_kernel(T* intensity_io, T c_val, size_t nxy)
	{
		size_t thr_idx = threadIdx.x + blockIdx.x*blockDim.x;
		size_t thr_num = blockDim.x*gridDim.x;
		while (thr_idx < nxy)
		{
			intensity_io[thr_idx] = c_val;
			thr_idx += thr_num;
		}
	}
	template <class T>
	__global__ void Fresenel_prop_kernel(Grid_2d_Dev<T> grid_2d, T w,
	T gxu, T gyu, T_complex<T>* wave_io)
	{
		int thr_idx = threadIdx.x + blockIdx.x*blockDim.x;
		int thr_idy = threadIdx.y + blockIdx.y*blockDim.y;

		if ((thr_idx < grid_2d.col) && (thr_idy < grid_2d.row))
		{
			const size_t ixy = grid_2d.row_major_idx(thr_idx, thr_idy);
			T amplitude = grid_2d.Grid_size_inverse();
			T theta = w*grid_2d.g_fft_shift_squared(thr_idx, thr_idy, gxu, gyu);
			if (grid_2d.bwl)
			{
				amplitude *= grid_2d.bwl_factor_shift(thr_idx, thr_idy);
			}
			T_complex<T> tempr;
			tempr.x = amplitude*cos(theta);
			tempr.y = amplitude*sin(theta);
			wave_io[ixy] = complex_Mul<T>(tempr, wave_io[ixy]);
		}
	}
	template <class T>
	__global__ void gen_propagator_kernel(Grid_2d_Dev<T> grid_2d, T w,
		T gxu, T gyu, T_complex<T>* prop_io)
	{
		int thr_idx = threadIdx.x + blockIdx.x*blockDim.x;
		int thr_idy = threadIdx.y + blockIdx.y*blockDim.y;

		if ((thr_idx < grid_2d.col) && (thr_idy < grid_2d.row))
		{
			const size_t ixy = grid_2d.row_major_idx(thr_idx, thr_idy);
			T amplitude = grid_2d.Grid_size_inverse();
			T theta = w*grid_2d.g_fft_shift_squared(thr_idx, thr_idy, gxu, gyu);
			if (grid_2d.bwl)
			{
				amplitude *= grid_2d.bwl_factor_shift(thr_idx, thr_idy);
			}
			prop_io[ixy].x = amplitude*cos(theta);
			prop_io[ixy].y = amplitude*sin(theta);
		}
	}
	template <class T>
	__global__ void	Fresenel_prop_batch_kernel(Grid_2d_Dev<T> grid_2d, T w,
		T gxu, T gyu, T_complex<T>* wave_io,int batch_num, size_t grid_size)
	{
		int thr_idx = threadIdx.x + blockIdx.x*blockDim.x;
		int thr_idy = threadIdx.y + blockIdx.y*blockDim.y;

		if ((thr_idx < grid_2d.col) && (thr_idy < grid_2d.row))
		{
			const size_t ixy = grid_2d.row_major_idx(thr_idx, thr_idy);
			T amplitude = grid_2d.Grid_size_inverse();
			T theta = w*grid_2d.g_fft_shift_squared(thr_idx, thr_idy, gxu, gyu);
			if (grid_2d.bwl)
			{
				amplitude *= grid_2d.bwl_factor_shift(thr_idx, thr_idy);
			}
			T_complex<T> tempr;
			tempr.x = amplitude*cos(theta);
			tempr.y = amplitude*sin(theta);
			for (auto i = 0; i < batch_num; i++)
			{
				size_t pix_idx = ixy + i*grid_size;
				wave_io[pix_idx] = complex_Mul<T>(tempr, wave_io[pix_idx]);
			}
		}
	}
	template <class T>
	__global__ void add_complex(T_complex<T>* wave_i, T_complex<T>* wave_o, size_t nxy)
	{
		size_t thr_idx = threadIdx.x + blockIdx.x*blockDim.x;
		size_t thr_num = blockDim.x*gridDim.x;
		while (thr_idx < nxy)
		{
			wave_o[thr_idx].x += wave_i[thr_idx].x;
			wave_o[thr_idx].y += wave_i[thr_idx].y;
			thr_idx += thr_num;
		}
	}
	template <class T>
	__global__ void wave_amplitude(T_complex<T>* wave_i, T* wave_o, size_t nxy)
	{
		size_t thr_idx = threadIdx.x + blockIdx.x*blockDim.x;
		size_t thr_num = blockDim.x*gridDim.x;
		while (thr_idx < nxy)
		{
			wave_o[thr_idx] = complex_Norm<T>(wave_i[thr_idx]);
			thr_idx += thr_num;
		}
	}
	template <class T>
	__global__ void wave_spliter(T_complex<T>* wave_i, T* real_o, T* imag_o, size_t nxy)
	{
		size_t thr_idx = threadIdx.x + blockIdx.x*blockDim.x;
		size_t thr_num = blockDim.x*gridDim.x;
		while (thr_idx < nxy)
		{
			real_o[thr_idx] = wave_i[thr_idx].x;
			imag_o[thr_idx] = wave_i[thr_idx].y;
			thr_idx += thr_num;
		}
	}
	template <class T>
	__global__ void add_scale_square(T w, T_complex<T>* wave_i, T* intensity_o, size_t nxy)
	{
		size_t thr_idx = threadIdx.x + blockIdx.x*blockDim.x;
		size_t thr_num = blockDim.x*gridDim.x;
		while (thr_idx < nxy)
		{
			intensity_o[thr_idx] += w*complex_Norm<T>(wave_i[thr_idx]);
			thr_idx += thr_num;
		}
	}
	template <class T>
	__global__ void add_scale_components(T w, T_complex<T>* wave_i, T* real_o, T* imag_o, size_t nxy)
	{
		size_t thr_idx = threadIdx.x + blockIdx.x*blockDim.x;
		size_t thr_num = blockDim.x*gridDim.x;
		while (thr_idx < nxy)
		{
			real_o[thr_idx] += w*wave_i[thr_idx].x;
			imag_o[thr_idx] += w*wave_i[thr_idx].y;
			thr_idx += thr_num;
		}
	}
	template <class T>
	__global__ void add_weighted_scale(T w, T* intensity_i, T* intensity_o, size_t nxy)
	{
		size_t thr_idx = threadIdx.x + blockIdx.x*blockDim.x;
		size_t thr_num = blockDim.x*gridDim.x;
		while (thr_idx < nxy)
		{
			intensity_o[thr_idx] += w*intensity_i[thr_idx];
			thr_idx += thr_num;
		}
	}
	template <class T>
	__global__ void scale_complex(T_complex<T>* wave_o, T scale_factor, size_t nxy)
	{
		size_t thr_idx = threadIdx.x + blockIdx.x*blockDim.x;
		size_t thr_num = blockDim.x*gridDim.x;
		while (thr_idx < nxy)
		{
			wave_o[thr_idx].x /= scale_factor;
			wave_o[thr_idx].y /= scale_factor;
			thr_idx += thr_num;
		}
	}
	template <class T>
	__global__ void fft_shift(Grid_2d_Dev<T> grid_2d, T* intensity_io)
	{
		int thr_idx = threadIdx.x + blockIdx.x*blockDim.x;
		int thr_idy = threadIdx.y + blockIdx.y*blockDim.y;
		if ((thr_idx < grid_2d.col_h) && (thr_idy < grid_2d.row_h))
		{
			size_t ixy = grid_2d.row_major_idx(thr_idx, thr_idy);
			size_t ixy_shifted = grid_2d.row_major_idx(thr_idx+grid_2d.col_h, thr_idy + grid_2d.row_h);
			//first and fourth
			T tempt = intensity_io[ixy];
			intensity_io[ixy] = intensity_io[ixy_shifted];
			intensity_io[ixy_shifted] = tempt;
			//second and third
			ixy= grid_2d.row_major_idx(thr_idx + grid_2d.col_h, thr_idy);
			ixy_shifted = grid_2d.row_major_idx(thr_idx, thr_idy+grid_2d.row_h);
			tempt = intensity_io[ixy];
			intensity_io[ixy] = intensity_io[ixy_shifted];
			intensity_io[ixy_shifted] = tempt;
		}
	}
	template <class T>
	__global__ void Generate_probe(Grid_2d_Dev<T> grid_2d, Lens_Dev<T> lens,
		T x, T y,T gxu, T gyu, T_complex<T>* probe_fun_o)
	{
		int thr_idx = threadIdx.x + blockIdx.x*blockDim.x;
		int thr_idy = threadIdx.y + blockIdx.y*blockDim.y;

		if ((thr_idx < grid_2d.col) && (thr_idy < grid_2d.row))
		{
			size_t ixy = grid_2d.row_major_idx(thr_idx, thr_idy);
			auto gx = grid_2d.gx_fft_shift(thr_idx) + gxu;
			auto gy = grid_2d.gy_fft_shift(thr_idy) + gyu;
			auto g2 = gx*gx + gy*gy;
			if ((lens.g2_min <= g2) && (g2 < lens.g2_max))
			{
				T phase_shift =coherent_phase_shift<T>(gx, gy, lens);
				phase_shift += (x*gx + y*gy);
				probe_fun_o[ixy] = euler(phase_shift);
			}
			else
			{
				probe_fun_o[ixy].x = T(0.0);
				probe_fun_o[ixy].y = T(0.0);
			}
		}
	}
	template <class T>
	__global__ void Filter_wave(Grid_2d_Dev<T> grid_2d, T g2_min, T g2_max, T_complex<T>* wave_io)
	{
		int thr_idx = threadIdx.x + blockIdx.x*blockDim.x;
		int thr_idy = threadIdx.y + blockIdx.y*blockDim.y;
		if ((thr_idx < grid_2d.col) && (thr_idy < grid_2d.row))
		{
			size_t ixy = grid_2d.row_major_idx(thr_idx, thr_idy);
			auto g2 = grid_2d.g_fft_shift_squared(thr_idx, thr_idy);
			if ((g2_min <= g2)&&(g2 <= g2_max))
			{
				T amplitude = grid_2d.Grid_size_inverse();
				wave_io[ixy].x *= amplitude;
				wave_io[ixy].y *= amplitude;
			}
			else
			{
				wave_io[ixy].x = T(0.0);
				wave_io[ixy].y = T(0.0);
			}
		}
	}
	template <class T>
	__global__ void Generate_multi_probe(Grid_2d_Dev<T> grid_2d, Lens_Dev<T> lens,
		T* x, T* y, T gxu, T gyu, int probe_num, size_t grid_size, T_complex<T>* probe_fun_o)
	{
		int thr_idx = threadIdx.x + blockIdx.x*blockDim.x;
		int thr_idy = threadIdx.y + blockIdx.y*blockDim.y;

		if ((thr_idx < grid_2d.col) && (thr_idy < grid_2d.row))
		{
			size_t ixy = grid_2d.row_major_idx(thr_idx, thr_idy);
			auto gx = grid_2d.gx_fft_shift(thr_idx) + gxu;
			auto gy = grid_2d.gy_fft_shift(thr_idy) + gyu;
			auto g2 = gx*gx + gy*gy;
			if ((lens.g2_min <= g2) && (g2 < lens.g2_max))
			{
				T phase_shift = coherent_phase_shift<T>(gx, gy, lens);
				for (auto i = 0; i < probe_num; i++)
				{
					size_t pix_idx = ixy + i*grid_size;
					auto phase_modulate = phase_shift + x[i] * gx + y[i] * gy;
					probe_fun_o[pix_idx] = euler(phase_modulate);
				}
			}
			else
			{
				for (auto i = 0; i < probe_num; i++)
				{
					size_t pix_idx = ixy + i*grid_size;
					probe_fun_o[pix_idx].x = T(0.0);
					probe_fun_o[pix_idx].y = T(0.0);
				}
			}
		}
	}
	template <class T>
	__global__ void apply_CTF_kernel(Grid_2d_Dev<T> grid_2d, Lens_Dev<T> lens, T gxu, T gyu, T_complex<T>* wave_i, T_complex<T>* wave_o)
	{
		int thr_idx = threadIdx.x + blockIdx.x*blockDim.x;
		int thr_idy = threadIdx.y + blockIdx.y*blockDim.y;

		if ((thr_idx < grid_2d.col) && (thr_idy < grid_2d.row))
		{
			size_t ixy = grid_2d.row_major_idx(thr_idx, thr_idy);
			auto gx = grid_2d.gx_fft_shift(thr_idx) + gxu;
			auto gy = grid_2d.gy_fft_shift(thr_idy) + gyu;
			auto g2 = gx*gx + gy*gy;
			if ((lens.g2_min <= g2) && (g2 < lens.g2_max))
			{
				T phase_shift = coherent_phase_shift<T>(gx, gy, lens);
				wave_o[ixy] = complex_Mul<T>(wave_i[ixy], euler(phase_shift));
			}
			else
			{
				wave_o[ixy].x = T(0.0);
				wave_o[ixy].y = T(0.0);
			}
		}
	}
	template <class T>
	__global__ void apply_CTF_kernel_batch(Grid_2d_Dev<T> grid_2d, Lens_Dev<T> lens, T gxu, T gyu,
		T* defocus, int batch_num, size_t grid_size, T_complex<T>* wave_i, T_complex<T>* wave_o)
	{
		int thr_idx = threadIdx.x + blockIdx.x*blockDim.x;
		int thr_idy = threadIdx.y + blockIdx.y*blockDim.y;

		if ((thr_idx < grid_2d.col) && (thr_idy < grid_2d.row))
		{
			size_t ixy = grid_2d.row_major_idx(thr_idx, thr_idy);
			auto gx = grid_2d.gx_fft_shift(thr_idx) + gxu;
			auto gy = grid_2d.gy_fft_shift(thr_idy) + gyu;
			auto g2 = gx*gx + gy*gy;
			if ((lens.g2_min <= g2) && (g2 < lens.g2_max))
			{
				//get the phase shift right at the defocus.
				T phase_shift = coherent_phase_shift<T>(gx, gy, lens);
				for (auto i = 0; i < batch_num; i++)
				{
					size_t pix_idx = ixy + i*grid_size;
					auto phase_modulate = phase_shift+defocus[i]*g2;
					wave_o[pix_idx] = complex_Mul<T>(wave_i[pix_idx], euler(phase_modulate));
				}
			}
			else
			{
				for (auto i = 0; i < batch_num; i++)
				{
					size_t pix_idx = ixy + i*grid_size;
					wave_o[pix_idx].x = T(0.0);
					wave_o[pix_idx].y = T(0.0);
				}
			}
		}
	}
	template <class T>
	__global__ void apply_PCTF_kernel(Grid_2d_Dev<T> grid_2d, Lens_Dev<T> lens, T_complex<T>* wave_i, T_complex<T>* wave_o)
	{
		int thr_idx = threadIdx.x + blockIdx.x*blockDim.x;
		int thr_idy = threadIdx.y + blockIdx.y*blockDim.y;

		if ((thr_idx < grid_2d.col) && (thr_idy < grid_2d.row))
		{
			size_t ixy = grid_2d.row_major_idx(thr_idx, thr_idy);
			auto gx = grid_2d.gx_fft_shift(thr_idx);
			auto gy = grid_2d.gy_fft_shift(thr_idy);
			auto v =  incoherent_phase_shift<T>(gx, gy, lens);
			wave_o[ixy] = complex_Mul<T>(wave_i[ixy], v);
		}
	}
	/********  *****Real space phase modulation by wave incident************/
	template <class T>
	__global__ void exp_r_factor_2d(Grid_2d_Dev<T> grid_2d, T gx, T gy,
		T_complex<T>* wave_i, T_complex<T>* wave_o)
	{
		int thr_idx = threadIdx.x + blockIdx.x*blockDim.x;
		int thr_idy = threadIdx.y + blockIdx.y*blockDim.y;
		if ((thr_idx < grid_2d.col) && (thr_idy < grid_2d.row))
		{
			const size_t ixy = grid_2d.row_major_idx(thr_idx, thr_idy);
			const auto Rx = grid_2d.X_distance(thr_idx) - 0.5*grid_2d.x_l;
			const auto Ry = grid_2d.Y_distance(thr_idy) - 0.5*grid_2d.y_l;
			wave_o[ixy] = complex_Mul<T>(wave_i[ixy], euler(gx*Rx + gy*Ry));
		}
	}
	// sum over the detector
	template <class T>
	__global__ void sum_square_over_Det(Grid_2d_Dev<T> grid_2d, T g2_min, T g2_max, T_complex<T>* M_i, T* Mp_o)
	{
		//shared memory that wait for the writing.
		__shared__ T cache[c_thrnxny*c_thrnxny];
		int cacheIndex = threadIdx.x + threadIdx.y*blockDim.x;
		//thread dimension
		int thr_idx = threadIdx.x + blockIdx.x*blockDim.x;
		int thr_idy = threadIdx.y + blockIdx.y*blockDim.y;

		T sum = 0;
		int iy = thr_idy;
		while (iy < grid_2d.row)
		{
			int ix = thr_idx;
			while (ix < grid_2d.col)
			{
				auto g2 = grid_2d.g_fft_shift_squared(ix, iy);
				if ((g2_min <= g2) && (g2 <= g2_max))
				{
					size_t ixy = grid_2d.row_major_idx(ix, iy);
					sum += complex_Norm<T>(M_i[ixy]);
				}
				ix += blockDim.x*gridDim.x;
			}
			iy += blockDim.y*gridDim.y;
		}
		cache[cacheIndex] = sum;
		__syncthreads();

		int i = c_thrnxny*c_thrnxny / 2;
		while (i != 0)
		{
			if (cacheIndex < i)
			{
				cache[cacheIndex] += cache[cacheIndex + i];
			}
			__syncthreads();
			i /= 2;
		}
		if (cacheIndex == 0)
		{
			Mp_o[blockIdx.x + blockIdx.y*gridDim.x] = cache[0];
		}
	}
	template <class T>
	__global__ void sum_over_Det(Grid_2d_Dev<T> grid_2d, T g2_min, T g2_max, T* M_i, T* Mp_o)
	{
		//shared memory that wait for the writing.
		__shared__ T cache[c_thrnxny*c_thrnxny];
		int cacheIndex = threadIdx.x + threadIdx.y*blockDim.x;
		//thread dimension
		int thr_idx = threadIdx.x + blockIdx.x*blockDim.x;
		int thr_idy = threadIdx.y + blockIdx.y*blockDim.y;

		T sum = 0;
		int iy = thr_idy;
		while (iy < grid_2d.row)
		{
			int ix = thr_idx;
			while (ix < grid_2d.col)
			{
				auto g2 = grid_2d.g_fft_shift_squared(ix, iy);
				if ((g2_min <= g2) && (g2 <= g2_max))
				{
					size_t ixy = grid_2d.row_major_idx(ix, iy);
					sum += M_i[ixy];
				}
				ix += blockDim.x*gridDim.x;
			}
			iy += blockDim.y*gridDim.y;
		}
		cache[cacheIndex] = sum;
		__syncthreads();

		int i = c_thrnxny*c_thrnxny / 2;
		while (i != 0)
		{
			if (cacheIndex < i)
			{
				cache[cacheIndex] += cache[cacheIndex + i];
			}
			__syncthreads();
			i /= 2;
		}
		if (cacheIndex == 0)
		{
			Mp_o[blockIdx.x + blockIdx.y*gridDim.x] = cache[0];
		}
	}
	template <class T>
	__global__ void sum_square_over_Det_batch(Grid_2d_Dev<T> grid_2d, T g2_min, T g2_max, T_complex<T>* M_i, T* Mp_o, size_t grid_size)
	{
		//shared memory that wait for the writing.
		__shared__ T cache[c_thrnxny*c_thrnxny];
		int cacheIndex = threadIdx.x + threadIdx.y*blockDim.x;
		int img_idx = blockIdx.z;
		//thread dimension
		int thr_idx = threadIdx.x + blockIdx.x*blockDim.x;
		int thr_idy = threadIdx.y + blockIdx.y*blockDim.y;

		T sum = 0;
		int iy = thr_idy;
		while (iy < grid_2d.row)
		{
			int ix = thr_idx;
			while (ix < grid_2d.col)
			{
				auto g2 = grid_2d.g_fft_shift_squared(ix, iy);
				if ((g2_min <= g2) && (g2 <= g2_max))
				{
					size_t ixy = grid_2d.row_major_idx(ix, iy)+img_idx*grid_size;
					sum += complex_Norm<T>(M_i[ixy]);
				}
				ix += blockDim.x*gridDim.x;
			}
			iy += blockDim.y*gridDim.y;
		}
		cache[cacheIndex] = sum;
		__syncthreads();

		int i = c_thrnxny*c_thrnxny / 2;
		while (i != 0)
		{
			if (cacheIndex < i)
			{
				cache[cacheIndex] += cache[cacheIndex + i];
			}
			__syncthreads();
			i /= 2;
		}
		if (cacheIndex == 0)
		{
			int block_idx = blockIdx.x + (blockIdx.y+img_idx*gridDim.y)*gridDim.x;
			Mp_o[block_idx] = cache[0];
		}
	}
}
#endif