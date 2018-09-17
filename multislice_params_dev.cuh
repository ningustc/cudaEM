#ifndef MULTISLICE_PARAMS_DEV_H
#define MULTISLICE_PARAMS_DEV_H
#include <cuda.h>
#include <cuda_runtime.h>
#include "host_basic_def.hpp"
#include "device_basic_def.cuh"
#include "atom_coor_data.hpp"
#include "multislice_params.hpp"
#include "fft_dev.cuh"
#include "device_containers.cuh"
#include "grid_dev.cuh"
#include "lens_dev.cuh"

namespace cudaEM
{
	template<class T>
	class multislice_params_dev
	{
	public:
		/************************shared variables with host multislice params**************/

		eTEM_Sim_Type simulation_type; 						// 1. TEM, 2. STEM
		ePotential_Type potential_type;						// potential type
		bool enable_fine_slicing; 				// ePS_Planes = 1, ePS_dz_Proj = 2, ePS_dz_Sub = 3
		ePhonon_Model phonon_model; 						// 1: Still atom model, 2: Absorptive potential model, 3: Frozen phonon
		FP_Dim pn_dim; 										// Phonon dimensions
		eTemporal_Spatial_Incoh temp_spat_incoh; 			// 1: Spatial and temporal, 2: Temporal, 3: Spatial, 4 Coherent
		eIllumination_Model illumination_model; 			// 1: Partial coherent approximation, 2: transmission cross coefficient
		eScanning_Type      stem_scanning;					//
		bool        record_full_data;
		int pn_seed; 										//Random seed(frozen phonon)
		bool scan_enable;
		bool store_trans;
		/*******************Specimen settings********************/
		int nrot; 											// Total number of rotations
		r3d<T> spec_rot_u0; 								// unitary vector	
		r3d<T> spec_rot_center_p;
		/***************Incident beam settings*******************/
		eIncident_Wave_Type iw_type; 						//Incident wave type
		T theta; 											// incident tilt (in spherical coordinates) (rad)
		T phi; 												// incident tilt (in spherical coordinates) (rad)
		T incident_wave_x_pos;								// x position
		T incident_wave_y_pos; 								// y position
		/*************************GPU Index related parameters*******************************/
		int GPU_ID;
		eTask_Distribution_Type GPU_task_schedule;
		int fp_idx_start;
		int fp_idx_end;
		int idx_pos_start;
		int idx_pos_end;
		/************************STEM dimensional settings*****************************/
		int STEM_row;
		int STEM_col;
		T scanning_interval_x;
		T scanning_interval_y;
		/********************The value to be determined after input********************/
		T lamda_val;
		T sigma_val;
		T gama_val;
		/*****************necessary host classes and methods*************/
		Tile_Scanner<T> tile_scanner;
		General_Scanner<T> gene_scanner;
		Atom_Coord_Data<T> input_atoms;
		Quad_1d<T> temporal_quard;
		Quad_2d<T> spatial_quard;
		Detector<T> circle_detectors;
		std::vector<T> R_maximums;
		std::vector<T> blank_intensity;
		/*********advanced variables that used in the device functions****/
		dim3 Grd_Dim_Sum;
		dim3 Blk_Dim_1D, Grd_Dim_1D;
		dim3 Blk_Dim_2D, Grd_Dim_2D;
		Grid_2d_Dev<T> grid_dev;
		FFT_Dev<T> fft_trans_dev;
		FFT_Dev<T> fft_prop_dev;
		Lens_Dev<T> cond_lens_dev;
		Lens_Dev<T> obj_lens_dev;
		//advanced variables requires the device storage
		std::vector<Element_Coeffs_Dev<T>> element_types_dev;
		T* qz_dev;
		T* x_pos_exp;
		T* y_pos_exp;
		//default initialization method.
		multislice_params_dev(multislice_params<T> input_host_data, int dev_ID )
		{
			initial_from_host(input_host_data, dev_ID);
		}
		void initial_from_host(multislice_params<T> input_host_data, int dev_ID)
		{
			/*************Set the GPU ID first*************/
			GPU_ID = dev_ID;
			if (input_host_data.frozen_phonon_enable())
			{
				GPU_task_schedule = eTDT_Phonon;
				fp_idx_start = GPU_ID*input_host_data.pn_nconf / input_host_data.GPU_num+1;
				fp_idx_end = (GPU_ID + 1)*input_host_data.pn_nconf / input_host_data.GPU_num;
			}
			else if (input_host_data.scanning_enable())
			{
				//for non phonon condition
				GPU_task_schedule = eTDT_Scanning;
				idx_pos_start = GPU_ID*input_host_data.output_scan_pos_num() / input_host_data.GPU_num;
				idx_pos_end = (GPU_ID + 1)*input_host_data.output_scan_pos_num() / input_host_data.GPU_num;
			}
			else
			{
				GPU_task_schedule = eTDT_Single_GPU;
			}
			/*************Shared parameter with host*************/
			potential_type = input_host_data.potential_type;
			illumination_model = input_host_data.illumination_model;
			temp_spat_incoh = input_host_data.temporal_spatial_incoh;
			phonon_model = input_host_data.phonon_model;
			enable_fine_slicing = input_host_data.enable_fine_slicing;
			simulation_type = input_host_data.simulation_type;
			pn_dim = input_host_data.pn_dim;
			record_full_data = (input_host_data.data_record==eWave);
			pn_seed = input_host_data.output_phonn_seed();
			spec_rot_u0 = input_host_data.spec_rot_u0;
			spec_rot_center_p = input_host_data.spec_rot_center_p;
			nrot = input_host_data.nrot;
			theta = input_host_data.theta;
			phi = input_host_data.phi;
			incident_wave_x_pos = input_host_data.incident_wave_x_pos;
			incident_wave_y_pos = input_host_data.incident_wave_y_pos;
			//initial the host supported classes
			input_atoms = input_host_data.output_Atom_data();
			temporal_quard = input_host_data.output_qt();
			spatial_quard = input_host_data.output_qs();
			iw_type = input_host_data.output_incident_wave_type();
			sigma_val = input_host_data.output_sigma();
			gama_val = input_host_data.output_gama();
			lamda_val = input_host_data.output_lamda();
			std::vector<T>* R_max_ptr = input_host_data.output_R_Maximums();
			R_maximums.assign(R_max_ptr->begin(), R_max_ptr->end());
			blank_intensity = *input_host_data.output_blankIntensity();
			//initial the device supported classes.
			auto temp = input_host_data.output_grid();
			grid_dev = temp;
			if (input_host_data.scanning_enable())
			{
				scanning_interval_x = input_host_data.output_scan_interval_x();
				scanning_interval_y = input_host_data.output_scan_interval_y();
				stem_scanning = input_host_data.stem_scanning;
				if (stem_scanning==eTile_scanning)
				{
					tile_scanner = input_host_data.output_tile_Scanner();
					cudaMalloc((void**)&x_pos_exp, sizeof(T)*tile_scanner.tile_x_exp.size());
					cudaMalloc((void**)&y_pos_exp, sizeof(T)*tile_scanner.tile_y_exp.size());
					cudaMemcpy(x_pos_exp, tile_scanner.tile_x_exp.data(),
						sizeof(T)*tile_scanner.tile_x_exp.size(), cudaMemcpyHostToDevice);
					cudaMemcpy(y_pos_exp, tile_scanner.tile_y_exp.data(),
						sizeof(T)*tile_scanner.tile_y_exp.size(), cudaMemcpyHostToDevice);
				}
				else
				{
					gene_scanner = input_host_data.output_general_Scanner();
					//set the incident beam direction. 
					incident_wave_y_pos = T(0); incident_wave_x_pos = T(0);
				}
				STEM_col = input_host_data.output_STEM_Col();
				STEM_row = input_host_data.output_STEM_Row();
			}
			else
			{
				obj_lens_dev = input_host_data.obj_lens;
			}
			circle_detectors = input_host_data.detector;
			scan_enable = input_host_data.scanning_enable();
			store_trans = input_host_data.batch_enable();
			cond_lens_dev = input_host_data.cond_lens;
			//set the kernel running parameters.
			Grd_Dim_Sum = dim3(4, 4);
			Blk_Dim_1D = dim3(c_thrnxy);
			Grd_Dim_1D = dim3(c_grid_1D);
			Blk_Dim_2D = dim3(c_thrnxny, c_thrnxny);
			Grd_Dim_2D = grid_dev.cuda_grid_2D();
			//initial the storage from the host data
			std::vector<Element_Coeffs<T>> * element_type_host_ptr = input_host_data.export_element_array();
			int element_type_size = (*element_type_host_ptr).size();
			element_types_dev.resize(element_type_size);
			for (auto ielement_type = 0; ielement_type < element_type_size; ielement_type++)
			{
				element_types_dev[ielement_type].assign((*element_type_host_ptr)[ielement_type]);
			}
			Quad_1d<T>* qz_host_ptr = input_host_data.output_qz();
			std::vector<float> qz_temp(2 * c_nqz);
			for (auto i = 0; i < c_nqz; i++)
			{
				qz_temp[2 * i] = qz_host_ptr->x[i];
				qz_temp[2 * i + 1] = qz_host_ptr->w[i];
			}
			cudaMalloc((void**)&qz_dev, sizeof(T) * 2 * c_nqz);
			cudaMemcpy(qz_dev, qz_temp.data(), sizeof(T) * 2 * c_nqz, cudaMemcpyHostToDevice);
			//initial the FFT transformation object
			//the first one used for transmission, the second one is used for propagation.
			fft_trans_dev.create_plan_2d_many(grid_dev.col, grid_dev.row, c_Probe_Batch);
			fft_prop_dev.create_plan_2d_many(grid_dev.col, grid_dev.row, c_Probe_Batch);
		}
		/****************************codes running on host****************************/
		T gx_0() const
		{
			 return std::sin(theta)*std::cos(phi) / lamda_val;
		}
		T gy_0() const
		{
			return std::sin(theta)*std::sin(phi) / lamda_val;
		}
		T  get_weight() const
		{
			if (phonon_model == ePM_Frozen_Phonon)
			{
				return 1.0 / static_cast<T>(fp_idx_end- fp_idx_start+1);
			}
			return 1.0;
		}
		inline void set_iscan_beam_position(int index_val)
		{
			incident_wave_x_pos = tile_scanner.x[index_val];
			incident_wave_y_pos = tile_scanner.y[index_val];
		}
	};
}
#endif
