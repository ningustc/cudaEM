#ifndef SCAN_SIMULATOR_H
#define SCAN_SIMULATOR_H
#include <thread>
#include "wave_function.cuh"
#include "multislice_params.hpp"
#include "multislice_params_dev.cuh"
#include "output_multislice_data.hpp"
namespace cudaEM
{
	template <class T>
	inline void run_STEM_tile_simu(multislice_params_dev<T> *multislice_param_dev, Wave_Function<T>* wave_function,
		const Atom_Coord_Data<T> &atoms_static, output_Multi_Data<T> *output_multislice)
	{
		Tile_Scanner<T>* scanner_ptr = &multislice_param_dev->tile_scanner;
		Grid_2d_Dev<T>* grid_dev_ptr = &multislice_param_dev->grid_dev;
		Detector<T>* detector_ptr = &multislice_param_dev->circle_detectors;
		std::vector<T>* blankintensity_ptr = &multislice_param_dev->blank_intensity;
		/****************************initialize the output STEM data******************/
		T_complex<T>* batch_waves_ptr;
		cudaMalloc((void **)&batch_waves_ptr,
			grid_dev_ptr->Grid_size()*c_Probe_Batch * sizeof(T_complex<T>));
		/*****************************************************************************/
		int STEM_row = multislice_param_dev->STEM_row;
		int STEM_col = multislice_param_dev->STEM_col;
		int STEM_Size = STEM_col*STEM_row;
		std::cout << "image col: " << STEM_col << std::endl;
		std::cout << "image row: " << STEM_row << std::endl;
		std::cout << "grid size: " << grid_dev_ptr->x_l << std::endl;
		std::cout << "grid size: " << grid_dev_ptr->y_l << std::endl;
		int tile_num = scanner_ptr->tile_num;
		output_multislice->init(STEM_row, STEM_col, detector_ptr->size());
		output_multislice->interval_x = scanner_ptr->x_interval;
		output_multislice->interval_y = scanner_ptr->y_interval;
		output_multislice->total_intensity = (*blankintensity_ptr)[detector_ptr->size()];
		T w = multislice_param_dev->get_weight();
		for (int iph = multislice_param_dev->fp_idx_start; iph <= multislice_param_dev->fp_idx_end; iph++)
		{
			std::cout << "phonon configuration: " << iph << std::endl;
			//simulate the whole specimen by separating the specimen into different tiles
			Atom_Coord_Data<T> atoms_vibrated;
			if (grid_dev_ptr->bwl)
			{
				vibrate_atom_positions(atoms_static, atoms_vibrated,
					multislice_param_dev->pn_seed, iph, multislice_param_dev->pn_dim);
			}
			else
			{
				vibrate_crystal_position(atoms_static, atoms_vibrated,
					multislice_param_dev->pn_seed, iph, multislice_param_dev->pn_dim);
			}
			for (auto itile = 0; itile < tile_num; itile++)
			{
				std::cout << "tile index: " << itile << std::endl;
				int tilt_col_idx = itile % scanner_ptr->col_block;
				int tilt_row_idx = itile / scanner_ptr->col_block;
				Atom_Coord_Data<T> atom_cutted = atoms_vibrated;
				T x_start = scanner_ptr->x_tile_pos[itile];
				T x_end = x_start + grid_dev_ptr->x_l;
				T y_start = scanner_ptr->y_tile_pos[itile];
				T y_end = y_start + grid_dev_ptr->y_l;
				remove_atoms_outside_xy_range(atom_cutted, x_start - 2.0f,
					x_end + 2.0f, y_start - 2.0f, y_end + 2.0f);
				atom_cutted.x_l = grid_dev_ptr->x_l;
				atom_cutted.y_l = grid_dev_ptr->y_l;
				atom_cutted.shift_atoms(x_start, y_start);
				//if there is no atom in this tile.
				if (atom_cutted.size() == 0)
				{
					std::cout << "tile index: " << itile << " is empty, and no atom" << std::endl;
					for (int iDet = 0; iDet < detector_ptr->size(); iDet++)
					{
						for (auto i = 0; i < scanner_ptr->tile_pixel; i++)
						{
							int y_idx = i / scanner_ptr->col_tile;
							int x_idx = i % scanner_ptr->col_tile;
							x_idx += tilt_col_idx*scanner_ptr->col_tile;
							y_idx += tilt_row_idx*scanner_ptr->row_tile;
							int pixel_idx = iDet* STEM_Size + y_idx*STEM_col + x_idx;
							output_multislice->STEM_intensity[pixel_idx] = (*blankintensity_ptr)[iDet];
						}
					}
					continue;
				}
				/****************************************************************/
				wave_function->update_atom(&atom_cutted);
				wave_function->store_sample_trans();
				int iprobe = 0;
				while (iprobe < scanner_ptr->tile_pixel)
				{
					//the atom numbers, and i_atom is the starting index
					int n_probe = std::min(c_Probe_Batch, scanner_ptr->tile_pixel - iprobe);
					std::vector<T> ADF_itensity(n_probe);
					wave_function->set_incident_wave_batch(batch_waves_ptr, iprobe, n_probe);
					wave_function->wave_modulation_batch(batch_waves_ptr);
					for (int iDet = 0; iDet < detector_ptr->size(); iDet++)
					{
						wave_function->integrated_intensity_batch(w, iDet, batch_waves_ptr, n_probe, ADF_itensity);
						for (auto i = 0; i < n_probe; i++)
						{
							int y_idx = (i + iprobe) / scanner_ptr->col_tile;
							int x_idx = (i + iprobe) % scanner_ptr->col_tile;
							x_idx += tilt_col_idx*scanner_ptr->col_tile;
							y_idx += tilt_row_idx*scanner_ptr->row_tile;
							int pixel_idx = iDet* STEM_Size + y_idx*STEM_col + x_idx;
							output_multislice->STEM_intensity[pixel_idx] += ADF_itensity[i];
						}
					}
					iprobe += n_probe;
				}
			}
		}
		cudaFree(batch_waves_ptr);
	}
	template <class T>
	inline void run_STEM_general_simu(multislice_params_dev<T> *multislice_param_dev, Wave_Function<T>* wave_function,
		const Atom_Coord_Data<T> &atoms_static, output_Multi_Data<T> *output_multislice)
	{
		General_Scanner<T>* scanner_ptr = &multislice_param_dev->gene_scanner;
		Grid_2d_Dev<T>* grid_dev_ptr = &multislice_param_dev->grid_dev;
		Detector<T>* detector_ptr = &multislice_param_dev->circle_detectors;
		std::vector<T>* blankintensity_ptr = &multislice_param_dev->blank_intensity;
		/****************************initialize the output STEM data******************/	
		T_complex<T>* waves_ptr;
		cudaMalloc((void **)&waves_ptr, grid_dev_ptr->Grid_size() * sizeof(T_complex<T>));;
		T w = multislice_param_dev->get_weight();
		int scan_pos_num = scanner_ptr->scan_pixel_num;
		output_multislice->init(scanner_ptr->row, scanner_ptr->col, detector_ptr->size());
		output_multislice->interval_x = scanner_ptr->scan_interval;
		output_multislice->interval_y = scanner_ptr->scan_interval;
		output_multislice->total_intensity = (*blankintensity_ptr)[detector_ptr->size()];
		for (int iph = multislice_param_dev->fp_idx_start; iph <= multislice_param_dev->fp_idx_end; iph++)
		{
			std::cout << "phonon configuration: " << iph << std::endl;
			//simulate the whole specimen by separating the specimen into different tiles
			Atom_Coord_Data<T> atoms_vibrated;
			if (grid_dev_ptr->bwl)
			{
				vibrate_atom_positions(atoms_static, atoms_vibrated,
					multislice_param_dev->pn_seed, iph, multislice_param_dev->pn_dim);
			}
			else
			{
				vibrate_crystal_position(atoms_static, atoms_vibrated,
					multislice_param_dev->pn_seed, iph, multislice_param_dev->pn_dim);
			}
			for (auto ipos = 0; ipos < scan_pos_num; ipos++)
			{
				Atom_Coord_Data<T> atom_cutted = atoms_vibrated;
				T x_start = scanner_ptr->x_probe_pos[ipos] - grid_dev_ptr->x_l/2;
				T x_end = x_start + grid_dev_ptr->x_l;
				T y_start = scanner_ptr->y_probe_pos[ipos] - grid_dev_ptr->y_l / 2;
				T y_end = y_start + grid_dev_ptr->y_l;
				remove_atoms_outside_xy_range(atom_cutted, x_start - 2.0f,
					x_end + 2.0f, y_start - 2.0f, y_end + 2.0f);
				atom_cutted.x_l = grid_dev_ptr->x_l;
				atom_cutted.y_l = grid_dev_ptr->y_l;
				atom_cutted.shift_atoms(x_start, y_start);
				//if there is no atom in this position.
				if (atom_cutted.size() == 0)
				{
					std::cout << "position index: " << ipos << " is empty, and no atom" << std::endl;
					for (int iDet = 0; iDet < detector_ptr->size(); iDet++)
					{
						int pixel_idx = iDet* scan_pos_num + ipos;
						output_multislice->STEM_intensity[pixel_idx] = (*blankintensity_ptr)[iDet];
					}
					continue;
				}
				/****************************************************************/
				wave_function->update_atom(&atom_cutted);
				//no need to store the projected potential or the transmission function.
				wave_function->set_incident_wave(waves_ptr);
				wave_function->wave_modulation(waves_ptr);
				for (int iDet = 0; iDet < detector_ptr->size(); iDet++)
				{
					int pixel_idx = iDet* scan_pos_num + ipos;
					output_multislice->STEM_intensity[pixel_idx] =
						wave_function->integrated_intensity(w, iDet, waves_ptr);
				}
			}
		}
		cudaFree(waves_ptr);
	}
	template <class T>
	inline void run_STEM_simu(multislice_params<T> *input_multislice_host_i, int GPU_ID, output_Multi_Data<T> *output_multislice)
	{
		cudaSetDevice(GPU_ID);
		multislice_params_dev<T> multislice_param_dev(*input_multislice_host_i, GPU_ID);
		Atom_Coord_Data<T> atoms_static = multislice_param_dev.input_atoms;
		Wave_Function<T> wave_function;
		wave_function.set_input_data(&multislice_param_dev);
		if (multislice_param_dev.stem_scanning==eTile_scanning)
		{
			run_STEM_tile_simu(&multislice_param_dev, &wave_function, atoms_static, output_multislice);
		}
		else
		{
			run_STEM_general_simu(&multislice_param_dev, &wave_function, atoms_static, output_multislice);
		}
	}
	template <class T>
	class scan_simulator
	{
	public:
		void set_input_data(multislice_params<T> *input_multislice_host_i, output_Multi_Data<T> *output_mulData_i)
		{
			GPU_Num = input_multislice_host_i->GPU_num;
			multisliceparam_host_ptr = input_multislice_host_i;
			output_mulData_ptr = output_mulData_i;
		}
		void STEM_multiGPU()
		{
			if (GPU_Num == 1)
			{
				run_STEM_simu<T>(multisliceparam_host_ptr, 0, output_mulData_ptr);
			}
			else
			{
				std::vector<output_Multi_Data<T>> output_values(GPU_Num);
				std::vector<std::thread> multislice_thread(GPU_Num - 1);
				for (auto iGPU = 0; iGPU < GPU_Num - 1; iGPU++)
				{
					multislice_thread[iGPU] = std::thread(run_STEM_simu<T>, multisliceparam_host_ptr, iGPU, &output_values[iGPU]);
				}
				run_STEM_simu<T>(multisliceparam_host_ptr, GPU_Num - 1, &output_values[GPU_Num - 1]);
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
			sum_data->interval_x = output_values[0].interval_x;
			sum_data->interval_y = output_values[0].interval_y;
			sum_data->total_intensity = output_values[0].total_intensity;
			int STEM_size = output_values[0].STEM_intensity.size();
			sum_data->STEM_intensity.resize(STEM_size, 0.0);
			for (auto iGPU = 0; iGPU < GPU_Num; iGPU++)
			{
				for (auto itensity = 0; itensity < STEM_size; itensity++)
				{
					sum_data->STEM_intensity[itensity] +=
						output_values[iGPU].STEM_intensity[itensity] / GPU_Num;
				}
			}
		}
		int GPU_Num;
		multislice_params<T>* multisliceparam_host_ptr;
	};
}
#endif