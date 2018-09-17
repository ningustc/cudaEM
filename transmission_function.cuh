#ifndef TRANSMISSION_FUNCTION_H
#define TRANSMISSION_FUNCTION_H
#include "device_basic_def.cuh"
#include "projected_potential.cuh"
#include "memory_manager.hpp"
#include "fftcallbacks.cuh"
namespace cudaEM
{
	template <class T>
	class Transmission_Function : public Projected_Potential<T>
	{
	public:

		Transmission_Function() : Projected_Potential<T>(), fft_2d_ptr(nullptr) 
		{
			//apply the binding of FFT and callback function.
			FFT_callback_type = get_callback_type<T>();
			if (std::is_same<T, float>::value)
			{
				cudaMemcpyFromSymbol(&callback_host_batch_ptr, d_loadCBCPtr_RB,
					sizeof(callback_host_batch_ptr));
			}
			else
			{
				cudaMemcpyFromSymbol(&callback_host_batch_ptr, d_loadCBZPtr_RB,
					sizeof(callback_host_batch_ptr));
			}
			if (std::is_same<T, float>::value)
			{
				cudaMemcpyFromSymbol(&callback_host_ptr, d_loadCBCPtr_R,
					sizeof(callback_host_ptr));
			}
			else
			{
				cudaMemcpyFromSymbol(&callback_host_ptr, d_loadCBZPtr_R,
					sizeof(callback_host_ptr));
			}
		}
		~Transmission_Function()
		{
			cudaFree(V_0_ptr);
			if (store_enable)
			{
				cudaFree(stored_potent_ptr);
			}
		}
		void set_input_data(multislice_params_dev<T> *input_multislice_i)
		{
			Projected_Potential<T>::set_input_data(input_multislice_i);
			fft_2d_ptr = &input_multislice_i->fft_trans_dev;
			/*****************Set the total pixel number***********************/
			Blk_Dim_1D = input_multislice_i->Blk_Dim_1D;
			Grd_Dim_1D = input_multislice_i->Grd_Dim_1D;
			Blk_Dim_2D = input_multislice_i->Blk_Dim_2D;
			Grd_Dim_2D = input_multislice_i->Grd_Dim_2D;
			grid_dev_ptr = &input_multislice_i->grid_dev;
			pixel_num = grid_dev_ptr->Grid_size();
			store_enable = input_multislice_i->store_trans;
			cudaMalloc((void**)&V_0_ptr, sizeof(T)*pixel_num);
			//set the size of the default transmission function.
			Vr_factor = get_Vr_factor(input_multislice_i->lamda_val, 
				input_multislice_i->gama_val, input_multislice_i->theta);
			//In TEM mode, the slicing is not stored, only in STEM mode, the slice will be stored
			//and will be reused for the computation of different probe position.
			if (!store_enable)
			{
				memory_slice.clear();
				return;
			}
			//set the memorized slice information
			int n_slice_sig = (int)std::ceil(3.0*this->atoms.sigma_max / grid_dev_ptr->dz);
			int n_slice_req = this->slicing.slices.size() + 4 * n_slice_sig;
			memory_slice.set_input_data(n_slice_req, pixel_num);
			cudaMalloc((void**)&stored_potent_ptr, sizeof(T)*pixel_num*memory_slice.n_slice_Allow);
		}
		void generate_trans(T w, T* potential_dev_ptr, T_complex<T> *Trans_dev_ptr)
		{
			generate_trans_function<T> << < Grd_Dim_1D, Blk_Dim_1D >> > (w, potential_dev_ptr, Trans_dev_ptr, pixel_num);
		}
		void get_trans_for_slices(const int &islice_0, const int &islice_e, T_complex<T> * Trans_dev_ptr)
		{
			Projected_Potential<T>::generate_potential_by_slice(islice_0, islice_e, V_0_ptr);
			generate_trans(Vr_factor, V_0_ptr, Trans_dev_ptr);
		}
		void get_trans_single_slice(const int &islice, T_complex<T> * Trans_dev_ptr)
		{
			Projected_Potential<T>::generate_potential_single_slice(islice, this->V_0_ptr);
			generate_trans(Vr_factor, V_0_ptr, Trans_dev_ptr);
		}
		void store_sample_trans()
		{
			if (!store_enable)
			{
				return;
			}
			for (auto islice = 0; islice < memory_slice.n_slice_cur(this->slicing.slices.size()); islice++)
			{
				potent_ptr = stored_potent_ptr+islice*pixel_num;
				Projected_Potential<T>::generate_potential_single_slice(islice, potent_ptr, Vr_factor);
			}
		}
		void attach_callback()
		{
			fft_2d_ptr->attach_callback((void**)&callback_host_ptr,
				FFT_callback_type, (void**)&V_0_ptr);
		}
		void transmit(const int &islice, T_complex<T>* wave_ptr)
		{
			Projected_Potential<T>::generate_potential_single_slice(islice, V_0_ptr, Vr_factor);
			fft_2d_ptr->forward(wave_ptr);
		}
		void transmit_batch(const int &islice, T_complex<T>* wave_ptr)
		{
			get_stored_potential(islice);
			fft_2d_ptr->attach_callback_batch((void**)&callback_host_batch_ptr,
				FFT_callback_type, (void**)&potent_ptr);
			fft_2d_ptr->forward_batch(wave_ptr);
			fft_2d_ptr->deattach_callback_batch(FFT_callback_type);
		}
		void deattach_callback()
		{
			fft_2d_ptr->deattach_callback(FFT_callback_type);
		}
		T Vr_factor;
	private:
		void get_stored_potential(const int &islice)
		{
			if (islice < memory_slice.n_slice_cur(this->slicing.slices.size()))
			{
				potent_ptr = stored_potent_ptr + islice*pixel_num;
			}
			else
			{
				Projected_Potential<T>::generate_potential_single_slice(islice, V_0_ptr, Vr_factor);
				potent_ptr = V_0_ptr;
			}
		}
		/******************************************************/
		Memory_manager<T> memory_slice;
		std::size_t pixel_num;
		Grid_2d_Dev<T>* grid_dev_ptr;
		bool store_enable;
		T* potent_ptr;
		T* stored_potent_ptr;
	protected:
		/******************************************************/
		dim3 Blk_Dim_1D, Grd_Dim_1D;
		dim3 Blk_Dim_2D, Grd_Dim_2D;
		FFT_Dev<T> *fft_2d_ptr;
		T* V_0_ptr;
		cufftXtCallbackType_t FFT_callback_type;
		T_Callback<T> callback_host_batch_ptr, callback_host_ptr;
	};
}
#endif