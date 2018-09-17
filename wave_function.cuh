#ifndef WAVE_FUNCTION_H
#define WAVE_FUNCTION_H
#include "multislice_params_dev.cuh"
#include "transmission_function.cuh"
#include "propagation_function.cuh"
#include "lens_modulation.cuh"
#include "incident_wave.cuh"
namespace cudaEM
{
	template <class T>
	class Wave_Function : public Transmission_Function<T>
	{
	public:
		Wave_Function() : Transmission_Function<T>() {}
		~Wave_Function()
		{
			cudaFree(intensity_dev0_ptr);
			cudaFree(imag_dev0_ptr);
			cudaFree(sum_ptr);
		}
		void set_input_data(multislice_params_dev<T> *input_multislice_i)
		{
			Transmission_Function<T>::set_input_data(input_multislice_i);
			multisliceparam_ptr = input_multislice_i;
			grid_dev_ptr = &input_multislice_i->grid_dev;
			fft_2d_ptr = &input_multislice_i->fft_trans_dev;
			detector_ptr = &input_multislice_i->circle_detectors;
			Blk_Dim_2D = input_multislice_i->Blk_Dim_2D;
			Grd_Dim_2D = input_multislice_i->Grd_Dim_2D;
			Blk_Dim_1D = input_multislice_i->Blk_Dim_1D;
			Grd_Dim_1D = input_multislice_i->Grd_Dim_1D;
			pixel_num =	 grid_dev_ptr->Grid_size();
			/*************************************************/
			cudaMalloc((void **)&intensity_dev0_ptr, sizeof(T)*pixel_num);
			cudaMalloc((void **)&imag_dev0_ptr, sizeof(T)*pixel_num);
			cudaMalloc((void **)&sum_ptr, sizeof(T)*c_Probe_Batch * 16);
			//initial the assist class to compute the exit wave.
			incident_wave.set_input_data(input_multislice_i);
			propagator.set_input_data(input_multislice_i);
			lens_modulator.set_input_data(input_multislice_i);
			/***************************************************/
			T gx_0 = multisliceparam_ptr->gx_0();
			T gy_0 = multisliceparam_ptr->gy_0();
			propagator.update_prop_function(gx_0, gy_0, grid_dev_ptr->dz);
		}
		/****************First step is generate the incident beam0***********/
		void set_incident_wave(T_complex<T> *wave_dev_ptr)
		{
			auto beam_x = multisliceparam_ptr->incident_wave_x_pos;
			auto beam_y = multisliceparam_ptr->incident_wave_y_pos;
			auto z_init = this->slicing.slice_middle_z(0);
			incident_wave.generate_incident_beam(wave_dev_ptr, 0, 0, beam_x, beam_y, z_init);
		}
		void set_incident_wave_batch(T_complex<T> * wave_dev_ptr, int start_idx, int batch_num)
		{
			auto z_init = this->slicing.slice_middle_z(0);
			incident_wave.generate_incident_beams(wave_dev_ptr, 0, 0, start_idx, batch_num, z_init);
		}
		/*****************simulate the wave transfer in the specimen********************/
		void wave_modulation(T_complex<T> * wave_dev_ptr)
		{
			//the input wave is in real space
			std::size_t islice;
			this->attach_callback();
			for (islice = 0; islice < this->slicing.slices.size()-1; islice++)
			{
				this->transmit(islice, wave_dev_ptr);
				propagator.propagate(wave_dev_ptr);
			}
			//Fourier transform the image into the Fourier space
			this->transmit(islice, wave_dev_ptr);
			this->deattach_callback();
		}
		void wave_modulation_batch(T_complex<T> * wave_dev_ptr)
		{
			std::size_t islice;
			//the input wave is in real space
			for (islice = 0; islice < this->slicing.slices.size()-1; islice++)
			{
				//real space transmission
				this->transmit_batch(islice, wave_dev_ptr);
				//Fourier space propagation, the input wave much be the real space wave
				propagator.propagate_batch(wave_dev_ptr);
			}
			//real space transmission
			this->transmit_batch(islice, wave_dev_ptr);
		}
		void single_slice_modulation(const int &islice, T_complex<T> * wave_dev_ptr)
		{
			//real space transmission
			this->transmit(islice, wave_dev_ptr);
			//Fourier space propagation, the input wave much be the real space wave
			propagator.propagate(wave_dev_ptr);
		}
		/*******************modulate the generated exit wave*******************/
		void OL_modulation(T weight, T_complex<T> * wave_dev_ptr, T* intensity_ptr)
		{
			//the input wave is in the reciprocal space.
			lens_modulator(wave_dev_ptr, intensity_dev0_ptr);
			//weight add the intensity to the output intensity
			add_weighted_scale<T> <<<Grd_Dim_1D, Blk_Dim_1D>>>(weight, intensity_dev0_ptr, intensity_ptr, pixel_num);
		}
		void OL_modulation(T weight, T_complex<T> * wave_dev_ptr, T* real_ptr, T* imag_ptr)
		{
			//the input wave is in the reciprocal space.
			lens_modulator(wave_dev_ptr, intensity_dev0_ptr, imag_dev0_ptr);
			//weight add the intensity to the output intensity
			add_weighted_scale<T> << <Grd_Dim_1D, Blk_Dim_1D >> >(weight, intensity_dev0_ptr, real_ptr, pixel_num);
			add_weighted_scale<T> << <Grd_Dim_1D, Blk_Dim_1D >> >(weight, imag_dev0_ptr, imag_ptr, pixel_num);
		}
		void generate_intensity(T_complex<T>* wave_dev_ptr, T* intensity_ptr)
		{
			wave_amplitude<T> << <Grd_Dim_1D, Blk_Dim_1D >> >(wave_dev_ptr, intensity_ptr, pixel_num);
		}
		void generate_intensity(T w_i, T_complex<T>* wave_dev_ptr, T* intensity_ptr)
		{
			add_scale_square<T> << <Grd_Dim_1D, Blk_Dim_1D >> >(w_i, wave_dev_ptr, intensity_ptr, pixel_num);
		}
		void generate_fullwave(T_complex<T>* wave_dev_ptr, T* real_ptr, T* imag_ptr)
		{
			wave_spliter<T> << <Grd_Dim_1D, Blk_Dim_1D >> >(wave_dev_ptr, real_ptr, imag_ptr, pixel_num);
		}
		void generate_fullwave(T w_i, T_complex<T>* wave_dev_ptr, T* real_ptr, T* imag_ptr)
		{
			add_scale_components<T> << <Grd_Dim_1D, Blk_Dim_1D >> >(w_i, wave_dev_ptr, real_ptr, imag_ptr, pixel_num);
		}
		void fft_shift_2d(T* intensity_ptr)
		{
			fft_shift<T> << <Grd_Dim_2D, Blk_Dim_2D >> >(*grid_dev_ptr, intensity_ptr);
		}
		void generate_intensity_batch(T_complex<T>* wave_dev_ptr, T* intensity_ptr, int batch_num)
		{
			wave_amplitude<T> << <Grd_Dim_1D, Blk_Dim_1D >> >(wave_dev_ptr, intensity_ptr, pixel_num*batch_num);
		}
		T integrated_intensity(T w_i, const int &iDet, T_complex<T>* wave_dev_ptr)
		{
			auto g_inner = detector_ptr->g_inner[iDet];
			auto g_outer = detector_ptr->g_outer[iDet];
			g_inner = std::pow(g_inner, 2);
			g_outer = std::pow(g_outer, 2);
			dim3 Grd_Dim_Sum(4, 4);
			sum_square_over_Det<T> << <Grd_Dim_Sum, Blk_Dim_2D >> >(*grid_dev_ptr, g_inner, g_outer, wave_dev_ptr, sum_ptr);
			std::vector<float> sums_host(16);
			cudaMemcpy(sums_host.data(), sum_ptr, sizeof(T)* 16, cudaMemcpyDeviceToHost);
			return w_i*std::accumulate(sums_host.begin(), sums_host.end(),T(0.0));
		}
		void integrated_intensity_batch(T w_i, const int &iDet, T_complex<T>* wave_dev_ptr, int batch_num, std::vector<T> &sums)
		{
			auto g_inner = detector_ptr->g_inner[iDet];
			auto g_outer = detector_ptr->g_outer[iDet];
			g_inner = std::pow(g_inner, 2);
			g_outer = std::pow(g_outer, 2);
			dim3 Grd_Dim_Sum(4, 4, batch_num);
			sum_square_over_Det_batch<T> << <Grd_Dim_Sum, Blk_Dim_2D >> >(*grid_dev_ptr, g_inner, g_outer, wave_dev_ptr, sum_ptr, pixel_num);
			sums.resize(batch_num);
			std::vector<float> sums_host(batch_num*16);
			cudaMemcpy(sums_host.data(), sum_ptr, sizeof(T)*batch_num * 16, cudaMemcpyDeviceToHost);
			for (int i = 0; i < batch_num; i++)
			{
				sums[i] = w_i*std::accumulate(sums_host.begin()+i*16, sums_host.begin() + (i+1) * 16, T(0.0));
			}
		}
		void wave_filter(const int &iDet, T_complex<T>* wave_dev_ptr)
		{
			auto g_inner = detector_ptr->g_inner[iDet];
			auto g_outer = detector_ptr->g_outer[iDet];
			//apply the forward Fourier transform to wave function. 
			fft_2d_ptr->forward(wave_dev_ptr);
			auto g2_inner = std::pow(g_inner, 2);
			auto g2_outer = std::pow(g_outer, 2);
			Filter_wave<T> << <Grd_Dim_2D, Blk_Dim_2D>> >(*grid_dev_ptr, g2_inner, g2_outer, wave_dev_ptr);
			//apply the inversed Fourier transform to the wave function.
			fft_2d_ptr->inverse(wave_dev_ptr);
		}
		T* intensity_dev0_ptr;
		T* imag_dev0_ptr;
		T* sum_ptr;
		Propagator<T> propagator;
		Lens_Modulation<T> lens_modulator;
		Incident_Wave<T> incident_wave;
		multislice_params_dev<T>* multisliceparam_ptr;
	private:
		Grid_2d_Dev<T>* grid_dev_ptr;
		dim3 Blk_Dim_2D, Grd_Dim_2D;
		dim3 Blk_Dim_1D, Grd_Dim_1D;
		FFT_Dev<T> *fft_2d_ptr;
		Detector<T>* detector_ptr;
		size_t pixel_num;
	};

}

#endif