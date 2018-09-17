#ifndef PROPAGATION_FUNCTION_H
#define PROPAGATION_FUNCTION_H
#include "host_basic_def.hpp"
#include "device_basic_def.cuh"
#include "multislice_params_dev.cuh"
#include "fftcallbacks.cuh"
namespace cudaEM
{
	template <class T>
	class Propagator {
	public:
		Propagator() : grid_dev_ptr(nullptr), fft_2d_ptr(nullptr) 
		{
			//apply the binding of FFT and callback function.
			FFT_callback_type = get_callback_type<T>();
		}
		~Propagator()
		{
			cudaFree(prop_dev0_ptr);
			fft_2d_ptr->deattach_callback(FFT_callback_type);
			fft_2d_ptr->deattach_callback_batch(FFT_callback_type);
		}
		void set_input_data(multislice_params_dev<T> *input_multislice_i)
		{
			fft_2d_ptr = &input_multislice_i->fft_prop_dev;
			grid_dev_ptr=&input_multislice_i->grid_dev;
			Grd_Dim_2D = input_multislice_i->Grd_Dim_2D;
			Blk_Dim_2D = input_multislice_i->Blk_Dim_2D;
			Grd_Dim_1D = input_multislice_i->Grd_Dim_1D;
			Blk_Dim_2D = input_multislice_i->Blk_Dim_2D;
			propagator_factor = -c_Pi*input_multislice_i->lamda_val;
			pixel_num = grid_dev_ptr->Grid_size();
			unit_wave_val.x = T(1.0); unit_wave_val.y = T(0.0);
			cudaMalloc((void**)&prop_dev0_ptr, pixel_num * sizeof(T_complex<T>));
			attach_callback();
			attach_callback_batch();
		}
		/************************************************************************/
		/* gxu and gyu is the tilting angle of incident beam, and z is thickness*/
		/************************************************************************/
		void propagate(T_complex<T> *wave_io_ptr)
		{
			fft_2d_ptr->inverse(wave_io_ptr);
		}
		void propagate_batch(T_complex<T> *wave_io_ptr)
		{
			fft_2d_ptr->inverse_batch(wave_io_ptr);
		}
		void update_prop_function(T gxu, T gyu, T z)
		{
			if (isZero(z))
			{
				set_wave_val_kernel<T> << <Grd_Dim_1D, Blk_Dim_1D >> > (prop_dev0_ptr, unit_wave_val, pixel_num);
				return;
			}
			gen_propagator_kernel<T> << <Grd_Dim_2D, Blk_Dim_2D >> > (*grid_dev_ptr, z*propagator_factor, gxu, gyu, prop_dev0_ptr);
		}
	private:
		void attach_callback()
		{
			T_Callback<T> callback_host_ptr;
			if (std::is_same<T, float>::value)
			{
				cudaMemcpyFromSymbol(&callback_host_ptr, d_loadCBCPtr_C,
					sizeof(callback_host_ptr));
			}
			else
			{
				cudaMemcpyFromSymbol(&callback_host_ptr, d_loadCBZPtr_C,
					sizeof(callback_host_ptr));
			}
			fft_2d_ptr->attach_callback((void**)&callback_host_ptr,
				FFT_callback_type, (void**)&prop_dev0_ptr);
		}
		void attach_callback_batch()
		{
			T_Callback<T> callback_host_ptr;
			if (std::is_same<T, float>::value)
			{
				cudaMemcpyFromSymbol(&callback_host_ptr, d_loadCBCPtr_CB,
					sizeof(callback_host_ptr));
			}
			else
			{
				cudaMemcpyFromSymbol(&callback_host_ptr, d_loadCBZPtr_CB,
					sizeof(callback_host_ptr));
			}
			fft_2d_ptr->attach_callback_batch((void**)&callback_host_ptr,
				FFT_callback_type, (void**)&prop_dev0_ptr);
		}
		dim3 Blk_Dim_2D, Grd_Dim_2D;
		dim3 Blk_Dim_1D, Grd_Dim_1D;
		Grid_2d_Dev<T>* grid_dev_ptr;
		T propagator_factor;
		FFT_Dev<T> *fft_2d_ptr;
		size_t pixel_num;
		T_complex<T>* prop_dev0_ptr;
		T_complex<T>  unit_wave_val;
		cufftXtCallbackType_t FFT_callback_type;
	};
}

#endif