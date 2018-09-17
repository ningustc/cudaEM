#ifndef THICKNESS_SIMULATOR_H
#define THICKNESS_SIMULATOR_H
#include <thread>
#include "wave_function.cuh"
#include "multislice_params.hpp"
#include "multislice_params_dev.cuh"
#include "output_multislice_data.hpp"
namespace cudaEM
{
	template <class T>
	inline void run_thickness_simu(multislice_params<T> *input_multislice_host_i, int GPU_ID, output_Multi_Data<T> *output_multislice)
	{
		cudaSetDevice(GPU_ID);
		multislice_params_dev<T> multislice_param_dev(*input_multislice_host_i, GPU_ID);
		Wave_Function<T> wave_function;
		wave_function.set_input_data(&multislice_param_dev);
		Grid_2d_Dev<T>* grid_dev_ptr = &multislice_param_dev.grid_dev;
		std::size_t pixel_num = grid_dev_ptr->Grid_size();
		std::size_t slice_num = wave_function.slicing.slices.size();
		//initialize the output data and initialize the device data
		output_multislice->row = grid_dev_ptr->row;
		output_multislice->col = grid_dev_ptr->col;
		output_multislice->thickness_frames.resize(slice_num*pixel_num);
		//initialize the data, allocate the space for the data
		T_complex<T>* wave_dev_ptr;
		T_complex<T>* wave_dev_temp_ptr;
		T* intensity_dev_ptr;
		cudaMalloc((void **)&wave_dev_ptr, pixel_num * sizeof(T_complex<T>));
		cudaMalloc((void **)&wave_dev_temp_ptr, pixel_num * sizeof(T_complex<T>));
		cudaMalloc((void **)&intensity_dev_ptr, pixel_num * sizeof(T)*slice_num);
		//propagate the wave along the thickness direction.
		if (multislice_param_dev.phonon_model != ePM_Frozen_Phonon)
		{
			wave_function.set_incident_wave(wave_dev_ptr);
			for (auto islice=0; islice <slice_num ;islice++)
			{
				wave_function.single_slice_modulation(islice, wave_dev_ptr);
				cudaMemcpy(wave_dev_temp_ptr, wave_dev_ptr, 
					sizeof(T)*pixel_num, cudaMemcpyDeviceToDevice);
				wave_function.wave_filter(1, wave_dev_temp_ptr);
				wave_function.generate_intensity(wave_dev_temp_ptr, intensity_dev_ptr);
				//copy the data from the intensity to 
				T* itensity_ptr_host = &output_multislice->thickness_frames[islice*pixel_num];
				cudaMemcpy(itensity_ptr_host, intensity_dev_ptr, pixel_num * sizeof(T), cudaMemcpyDeviceToHost);
			}
		}
		else
		{
			cudaMemset(intensity_dev_ptr, 0, sizeof(T)*pixel_num*slice_num);
			T w = multislice_param_dev.get_weight();
			for (auto iconf = multislice_param_dev.fp_idx_start; iconf <= multislice_param_dev.fp_idx_end; iconf++)
			{
				std::cout << iconf << std::endl;
				//generated the vibrated atom coordinate
				Atom_Coord_Data<T> atoms_vibrated;
				if (grid_dev_ptr->bwl)
				{
					vibrate_atom_positions(multislice_param_dev.input_atoms, atoms_vibrated,
						multislice_param_dev.pn_seed, iconf, multislice_param_dev.pn_dim);
				}
				else
				{
					vibrate_crystal_position(multislice_param_dev.input_atoms, atoms_vibrated,
						multislice_param_dev.pn_seed, iconf, multislice_param_dev.pn_dim);
				}
				//input the atom coordinate and generate the intensity
				wave_function.update_atom(&atoms_vibrated);
				wave_function.set_incident_wave(wave_dev_ptr);
				for (auto islice = 0; islice < slice_num; islice++)
				{
					wave_function.single_slice_modulation(islice, wave_dev_ptr);
					T* intensity_temp_ptr = intensity_dev_ptr+islice*pixel_num;
					cudaMemcpy(wave_dev_temp_ptr, wave_dev_ptr, sizeof(T)*pixel_num, cudaMemcpyDeviceToDevice);
					wave_function.wave_filter(1, wave_dev_temp_ptr);
					wave_function.generate_intensity(w, wave_dev_temp_ptr, intensity_temp_ptr);
				}
			}
			//copy the data from device and host
			T* itensity_ptr_host = output_multislice->thickness_frames.data();
			cudaMemcpy(itensity_ptr_host, intensity_dev_ptr, pixel_num*slice_num * sizeof(T), cudaMemcpyDeviceToHost);
		}
		cudaFree(wave_dev_ptr);
		cudaFree(wave_dev_temp_ptr);
		cudaFree(intensity_dev_ptr);
	}
	template <class T>
	class Thickness_simulator
	{
	public:
		void set_input_data(multislice_params<T> *input_multislice_host_i, output_Multi_Data<T> *output_mulData_i)
		{
			GPU_Num = input_multislice_host_i->GPU_num;
			multisliceparam_host_ptr = input_multislice_host_i;
			output_mulData_ptr = output_mulData_i;
		}
		void Thickness_multiGPU()
		{
			if (GPU_Num==1)
			{
				run_thickness_simu<T>(multisliceparam_host_ptr, 0, output_mulData_ptr);
			}
			else
			{
				std::vector<output_Multi_Data<T>> output_values(GPU_Num);
				std::vector<std::thread> multislice_thread(GPU_Num - 1);
				for (auto iGPU = 0; iGPU < GPU_Num - 1; iGPU++)
				{
					multislice_thread[iGPU] = std::thread(run_thickness_simu<T>, multisliceparam_host_ptr, iGPU, &output_values[iGPU]);
				}
				run_thickness_simu<T>(multisliceparam_host_ptr, GPU_Num - 1, &output_values[GPU_Num - 1]);
				for (auto iGPU = 0; iGPU < GPU_Num - 1; iGPU++)
				{
					multislice_thread[iGPU].join();
				}
				//sum all the intensities.
				combine_data(output_mulData_ptr, output_values);
			}
		}
		output_Multi_Data<T> *output_mulData_ptr;
	private:
		void combine_data(output_Multi_Data<T>* sum_data, std::vector<output_Multi_Data<T>> &output_values)
		{
			sum_data->row = output_values[0].row;
			sum_data->col = output_values[0].col;
			int image_size = output_values[0].thickness_frames.size();
			sum_data->thickness_frames.resize(image_size, 0.0f);
			for (auto iGPU = 0; iGPU < GPU_Num; iGPU++)
			{
				for (auto itensity = 0; itensity < image_size; itensity++)
				{
					sum_data->thickness_frames[itensity] +=
						output_values[iGPU].thickness_frames[itensity] / GPU_Num;
				}
			}
		}
		int GPU_Num;
		multislice_params<T>* multisliceparam_host_ptr;
	};
}
#endif
