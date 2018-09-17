#ifndef TEM_SIMULATOR_H
#define TEM_SIMULATOR_H
#include <thread>
#include "wave_function.cuh"
#include "multislice_params.hpp"
#include "multislice_params_dev.cuh"
#include "output_multislice_data.hpp"
namespace cudaEM
{
	template <class T>
	inline void ED_HRTEM(multislice_params_dev<T> *input_mulData, Wave_Function<T>* wave_function,
		output_Multi_Data<T> &output_multislice, eSpace output_space)
	{
		Grid_2d_Dev<T>* grid_dev_ptr = &input_mulData->grid_dev;
		//initialize the output data and initialize the device data
		bool save_full_wave = input_mulData->record_full_data;
		output_multislice.init(grid_dev_ptr->row, grid_dev_ptr->col, 0, save_full_wave);
		std::size_t pixel_num = grid_dev_ptr->Grid_size();
		T_complex<T>* wave_dev_ptr;
		T* intensity_dev_ptr;
		T* imag_dev_ptr;
		cudaMalloc((void **)&wave_dev_ptr, pixel_num* sizeof(T_complex<T>));
		cudaMalloc((void **)&intensity_dev_ptr, pixel_num* sizeof(T));
		cudaMalloc((void **)&imag_dev_ptr, pixel_num * sizeof(T));
		/*****************************************************************/
		T w;
		if (input_mulData->phonon_model != ePM_Frozen_Phonon)
		{
			w = T(1.0);
			wave_function->set_incident_wave(wave_dev_ptr);
			wave_function->wave_modulation(wave_dev_ptr);
			if (output_space == eS_Real)
			{
				if (save_full_wave)
				{
					wave_function->OL_modulation(w, wave_dev_ptr, intensity_dev_ptr, imag_dev_ptr);
				}
				else
				{
					wave_function->OL_modulation(w, wave_dev_ptr, intensity_dev_ptr);
				}
			}
			else
			{
				if (save_full_wave)
				{
					wave_function->generate_fullwave(wave_dev_ptr, intensity_dev_ptr, imag_dev_ptr);
				}
				else
				{
					wave_function->generate_intensity(wave_dev_ptr, intensity_dev_ptr);
				}
			}
		}
		else
		{
			cudaMemset(intensity_dev_ptr, 0, sizeof(T)*pixel_num);
			cudaMemset(imag_dev_ptr, 0, sizeof(T)*pixel_num);
			w = input_mulData->get_weight();
			for (auto iconf = input_mulData->fp_idx_start; iconf <= input_mulData->fp_idx_end; iconf++)
			{
				std::cout << iconf << std::endl;
				//generated the vibrated atom coordinate
				Atom_Coord_Data<T> atoms_vibrated;
				if (grid_dev_ptr->bwl)
				{
					vibrate_atom_positions(input_mulData->input_atoms, atoms_vibrated,
						input_mulData->pn_seed, iconf, input_mulData->pn_dim);
				}
				else
				{
					vibrate_crystal_position(input_mulData->input_atoms, atoms_vibrated,
						input_mulData->pn_seed, iconf, input_mulData->pn_dim);
				}
				//input the atom coordinate and generate the intensity
				wave_function->update_atom(&atoms_vibrated);
				wave_function->set_incident_wave(wave_dev_ptr);
				wave_function->wave_modulation(wave_dev_ptr);
				if (output_space == eS_Real)
				{
					if (save_full_wave)
					{
						wave_function->OL_modulation(w, wave_dev_ptr, intensity_dev_ptr, imag_dev_ptr);
					}
					else
					{
						wave_function->OL_modulation(w, wave_dev_ptr, intensity_dev_ptr);
					}
				}
				else
				{
					if (save_full_wave)
					{
						wave_function->generate_fullwave(w, wave_dev_ptr, intensity_dev_ptr, imag_dev_ptr);
					}
					else
					{
						wave_function->generate_intensity(w, wave_dev_ptr, intensity_dev_ptr);
					}
				}
			}
		}
		if (output_space==eS_Reciprocal)
		{
			wave_function->fft_shift_2d(intensity_dev_ptr);
			if (save_full_wave)
			{
				wave_function->fft_shift_2d(imag_dev_ptr);
			}
		}
		if (save_full_wave)
		{
			T* real_ptr_host = output_multislice.image_real.data();
			T* imag_ptr_host = output_multislice.image_imag.data();
			cudaMemcpy(real_ptr_host, intensity_dev_ptr, pixel_num*sizeof(T), cudaMemcpyDeviceToHost);
			cudaMemcpy(imag_ptr_host, imag_dev_ptr, pixel_num*sizeof(T), cudaMemcpyDeviceToHost);
		}
		else
		{
			T* itensity_ptr_host = output_multislice.image_intensity.data();
			cudaMemcpy(itensity_ptr_host, intensity_dev_ptr, pixel_num*sizeof(T), cudaMemcpyDeviceToHost);
		}
		cudaFree(intensity_dev_ptr);
		cudaFree(wave_dev_ptr);
		cudaFree(imag_dev_ptr);
	}
	template <class T>
	inline void CBED_CBEI(multislice_params_dev<T> *input_mulData, Wave_Function<T>* wave_function,
		output_Multi_Data<T> &output_multislice)
	{
		Grid_2d_Dev<T>* grid_dev_ptr = &input_mulData->grid_dev;
		//initialize the output data and initialize the device data
		bool save_full_wave = input_mulData->record_full_data;
		output_multislice.init(grid_dev_ptr->row, grid_dev_ptr->col, 0, save_full_wave);
		std::size_t pixel_num = grid_dev_ptr->Grid_size();
		T_complex<T>* wave_dev_ptr;
		T* intensity_dev_ptr;
		T*imag_dev_ptr;
		cudaMalloc((void **)&wave_dev_ptr, pixel_num * sizeof(T_complex<T>));
		cudaMalloc((void **)&intensity_dev_ptr, pixel_num * sizeof(T));
		cudaMalloc((void **)&imag_dev_ptr, pixel_num * sizeof(T));
		if (input_mulData->phonon_model != ePM_Frozen_Phonon)
		{
			wave_function->set_incident_wave(wave_dev_ptr);
			wave_function->wave_modulation(wave_dev_ptr);
			if (save_full_wave)
			{
				wave_function->generate_fullwave(wave_dev_ptr, intensity_dev_ptr, imag_dev_ptr);
			}
			else
			{
				wave_function->generate_intensity(wave_dev_ptr, intensity_dev_ptr);
			}
		}
		else
		{
			cudaMemset(intensity_dev_ptr, 0, sizeof(T)*pixel_num);
			cudaMemset(imag_dev_ptr, 0, sizeof(T)*pixel_num);
			T w = input_mulData->get_weight();
			for (auto iconf = input_mulData->fp_idx_start; iconf <= input_mulData->fp_idx_end; iconf++)
			{
				std::cout << iconf << std::endl;
				//generated the vibrated atom coordinate
				Atom_Coord_Data<T> atoms_vibrated;
				if (grid_dev_ptr->bwl)
				{
					vibrate_atom_positions(input_mulData->input_atoms, atoms_vibrated,
						input_mulData->pn_seed, iconf, input_mulData->pn_dim);
				}
				else
				{
					vibrate_crystal_position(input_mulData->input_atoms, atoms_vibrated,
						input_mulData->pn_seed, iconf, input_mulData->pn_dim);
				}
				//input the atom coordinate and generate the intensity
				wave_function->update_atom(&atoms_vibrated);
				wave_function->set_incident_wave(wave_dev_ptr);
				wave_function->wave_modulation(wave_dev_ptr);
				if (save_full_wave)
				{
					wave_function->generate_fullwave(w, wave_dev_ptr, intensity_dev_ptr, imag_dev_ptr);
				}
				else
				{
					wave_function->generate_intensity(w, wave_dev_ptr, intensity_dev_ptr);;
				}
			}
		}
		wave_function->fft_shift_2d(intensity_dev_ptr);
		if (save_full_wave)
		{
			wave_function->fft_shift_2d(imag_dev_ptr);
		}
		if (save_full_wave)
		{
			T* real_ptr_host = output_multislice.image_real.data();
			T* imag_ptr_host = output_multislice.image_imag.data();
			cudaMemcpy(real_ptr_host, intensity_dev_ptr, pixel_num*sizeof(T), cudaMemcpyDeviceToHost);
			cudaMemcpy(imag_ptr_host, imag_dev_ptr, pixel_num*sizeof(T), cudaMemcpyDeviceToHost);
		}
		else
		{
			T* itensity_ptr_host = output_multislice.image_intensity.data();
			cudaMemcpy(itensity_ptr_host, intensity_dev_ptr, pixel_num*sizeof(T), cudaMemcpyDeviceToHost);
		}
	}
	template <class T>
	inline void run_simu(multislice_params<T> *input_multislice_host_i, int GPU_ID, output_Multi_Data<T> *output_multislice)
	{
		cudaSetDevice(GPU_ID);
		multislice_params_dev<T> multislice_param_dev(*input_multislice_host_i, GPU_ID);
		Wave_Function<T> wave_function;
		wave_function.set_input_data(&multislice_param_dev);
		
		if (multislice_param_dev.simulation_type == eTEMST_HRTEM)
		{
			ED_HRTEM(&multislice_param_dev, &wave_function, *output_multislice, eS_Real);
		}
		else if (multislice_param_dev.simulation_type == eTEMST_ED)
		{
			ED_HRTEM(&multislice_param_dev, &wave_function, *output_multislice, eS_Reciprocal);
		}
		else if (multislice_param_dev.simulation_type == eTEMST_CBED)
		{
			CBED_CBEI(&multislice_param_dev, &wave_function, *output_multislice);
		}
	}
	template <class T>
	class TEM_simulator
	{
	public:
		void set_input_data(multislice_params<T> *input_multislice_host_i, output_Multi_Data<T> *output_mulData_i)
		{
			GPU_Num = input_multislice_host_i->GPU_num;
			multisliceparam_host_ptr = input_multislice_host_i;
			output_mulData_ptr = output_mulData_i;
		}
		void TEM_multiGPU()
		{
			if (GPU_Num==1)
			{
				 run_simu<T>(multisliceparam_host_ptr, 0, output_mulData_ptr);
			}
			else
			{
				std::vector<output_Multi_Data<T>> output_values(GPU_Num);
				std::vector<std::thread> multislice_thread(GPU_Num - 1);
				for (auto iGPU = 0; iGPU < GPU_Num - 1; iGPU++)
				{
					multislice_thread[iGPU] = std::thread(run_simu<T>, multisliceparam_host_ptr, iGPU, &output_values[iGPU]);
				}
				run_simu<T>(multisliceparam_host_ptr, GPU_Num - 1, &output_values[GPU_Num - 1]);
				for (auto iGPU = 0; iGPU < GPU_Num - 1; iGPU++)
				{
					multislice_thread[iGPU].join();
				}
				//sum all the intensities.
				combine_data(output_mulData_ptr, output_values, multisliceparam_host_ptr->data_record== eWave);
			}
		}
		output_Multi_Data<T> *output_mulData_ptr;
	private:
		void combine_data(output_Multi_Data<T>* sum_data, std::vector<output_Multi_Data<T>> &output_values, bool full_wave)
		{
			sum_data->row = output_values[0].row;
			sum_data->col = output_values[0].col;
			if (full_wave)
			{
				int image_size = output_values[0].image_real.size();
				sum_data->image_imag.resize(image_size, 0.0f);
				sum_data->image_real.resize(image_size, 0.0f);
				for (auto iGPU = 0; iGPU < GPU_Num; iGPU++)
				{
					for (auto itensity = 0; itensity < image_size; itensity++)
					{
						sum_data->image_real[itensity] +=
							output_values[iGPU].image_real[itensity] / GPU_Num;
						sum_data->image_imag[itensity] +=
							output_values[iGPU].image_imag[itensity] / GPU_Num;
					}
				}
			}
			else
			{
				int image_size = output_values[0].image_intensity.size();
				sum_data->image_intensity.resize(image_size, 0.0f);
				for (auto iGPU = 0; iGPU < GPU_Num; iGPU++)
				{
					for (auto itensity = 0; itensity < image_size; itensity++)
					{
						sum_data->image_intensity[itensity] +=
							output_values[iGPU].image_intensity[itensity] / GPU_Num;
					}
				}
			}
			
		}
		int GPU_Num;
		multislice_params<T>* multisliceparam_host_ptr;
	};
}
#endif
