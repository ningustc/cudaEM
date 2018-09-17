#include "element_coeffs_loader.hpp"
#include <cstdlib>
#include <iostream>
#include "multislice_params.hpp"
#include "multislice_params_dev.cuh"
#include "single_simulator.cuh"
#include "scan_simulator.cuh"
#include "thickness_simulator.cuh"
#include "iolib.hpp"
#include "rawdata_io.hpp"
void direct_xyzreader(std::vector<cudaEM::Atom<float>> &atom_combined)
{
	xyzreader("Au.xyz", atom_combined);
	for (auto iatom=0; iatom <atom_combined.size(); iatom++)
	{
		atom_combined[iatom].occupation = 1.0;
		atom_combined[iatom].sigma = 0.085f;
	}
}
void phonon_builder(std::vector<cudaEM::Atom<float>> &atom_combined)
{
	atom_combined.resize(3);
	atom_combined[0].x = 0.0f;
	atom_combined[0].y = 0.0f;

	atom_combined[1].x = 3.0f;
	atom_combined[1].y = 3.0f;

	atom_combined[2].x = 6.0f;
	atom_combined[2].y = 6.0f;
	for (auto iatom = 0; iatom < atom_combined.size(); iatom++)
	{
		atom_combined[iatom].zIndex = 14;
		atom_combined[iatom].z = 0.0f;
		atom_combined[iatom].occupation = 1.0;
		atom_combined[iatom].sigma = 0.085f;
	}
}
void TEM()
{
	int row = 512;
	int col = 512;
	cudaEM::multislice_params<float> input_param;
	input_param.simulation_type = cudaEM::eTEMST_HRTEM;
	input_param.potential_type = cudaEM::ePT_Lobato_0_12;
	input_param.phonon_model = cudaEM::ePM_Frozen_Phonon;
	input_param.enable_fine_slicing = true;
	input_param.illumination_model = cudaEM::eIM_Trans_Cross_Coef;
	input_param.temporal_spatial_incoh = cudaEM::eTSI_Temporal_Spatial;
	/******************Phonon settings**********************/
	input_param.pn_nconf = 100;
	input_param.E_0 = 200;
	input_param.pn_dim = cudaEM::FP_Dim(true, true, true);
	/*******************Atom settings************************/
	direct_xyzreader(input_param.input_atom_coordinates);
	input_param.is_crystal = false;
	input_param.slice_interval = 0.5;
	input_param.specimen_ratation = false;
	input_param.blank_ratio = 1.0;
	/**********************Grid settings**************************/
	input_param.img_col = col;
	input_param.img_row = row;
	input_param.GPU_num = 1;
	/****************initialize the input params******************/
	input_param.obj_lens.c_10 = 0;
	input_param.obj_lens.outer_aper_ang = 0.0f;
	input_param.obj_lens.dsf_sigma = 30.0;
	input_param.obj_lens.ssf_beta = 0.001f;
	input_param.obj_lens.dsf_npoints = 128;
	input_param.obj_lens.ssf_npoints = 8;
	/************************************************************/
	input_param.validate_parameters();
	input_param.initial_OL();
	cudaEM::TEM_simulator<float> simulation;
	cudaEM::output_Multi_Data<float> output_data;
	simulation.set_input_data(&input_param, &output_data);
	std::cout << "finish initialization" << std::endl;
	simulation.TEM_multiGPU();
	cudaEM::EMdata_IO rawSaver("D:/gold.h5", "gold");
	rawSaver.create_file();
	rawSaver.write_TEM_data(row, col, output_data.image_intensity);
}
void Thickness()
{
	int row = 512;
	int col = 512;
	float scan_interval= 1.0;
	cudaEM::multislice_params<float> input_param;
	input_param.simulation_type = cudaEM::eTEMST_CBED;
	input_param.potential_type = cudaEM::ePT_Kirkland_0_12;
	input_param.phonon_model = cudaEM::ePM_Frozen_Phonon;
	input_param.enable_fine_slicing = false;
	input_param.illumination_model = cudaEM::eIM_Trans_Cross_Coef;
	input_param.temporal_spatial_incoh = cudaEM::eTSI_Temporal;
	/******************Phonon settings**********************/
	input_param.pn_nconf = 40;
	input_param.E_0 = 200;
	input_param.pn_dim = cudaEM::FP_Dim(true, true, true);
	/*******************Atom settings************************/
	direct_xyzreader(input_param.input_atom_coordinates);
	input_param.is_crystal = true;
	input_param.slice_interval = 0.5;
	input_param.specimen_ratation = false;
	/**********************Grid settings**************************/
	input_param.img_col = col;
	input_param.img_row = row;
	input_param.GPU_num = 2;
	/****************initialize the input params******************/
	input_param.cond_lens.c_10 = 0.0;
	input_param.cond_lens.c_30 = 0.0;
	input_param.cond_lens.inner_aper_ang = 0.0;
	input_param.cond_lens.outer_aper_ang = 0.030;
	input_param.cond_lens.dsf_sigma = 3.0;
	input_param.cond_lens.dsf_npoints = 16;
	input_param.detector.resize(2);
	input_param.detector.g_inner[0] = 15;
	input_param.detector.g_outer[0] = 30;
	input_param.detector.g_inner[1] = 100;
	input_param.detector.g_outer[1] = 250;
	/************************************************************/
	input_param.validate_parameters();
	input_param.initial_CL();
	cudaEM::Thickness_simulator<float> simulation;
	int angle_num=30;
	for (int iscan=0; iscan <angle_num; iscan++)
	{
		cudaEM::output_Multi_Data<float> output_data;
		//reset the position of the beam
		input_param.incident_wave_x_pos = scan_interval*std::sin(iscan*cudaEM::c_2Pi / angle_num);
		input_param.incident_wave_y_pos = scan_interval*std::cos(iscan*cudaEM::c_2Pi / angle_num);
		simulation.set_input_data(&input_param, &output_data);
		std::cout << "finish initialization" << std::endl;
		simulation.Thickness_multiGPU();
		//output the thickness series
		int slice_num = output_data.thickness_frames.size() / (row*col);
	}
}
void exitwave()
{
	int row = 1024;
	int col = 1024;
	cudaEM::multislice_params<float> input_param;
	input_param.simulation_type = cudaEM::eTEMST_HRTEM;
	input_param.potential_type = cudaEM::ePT_Lobato_0_12;
	input_param.phonon_model = cudaEM::ePM_Frozen_Phonon;
	input_param.enable_fine_slicing = false;
	input_param.illumination_model = cudaEM::eIM_Trans_Cross_Coef;
	input_param.temporal_spatial_incoh = cudaEM::eTSI_Temporal;
	//save the exit wave.
	input_param.data_record = cudaEM::eWave;
	/******************Phonon settings**********************/
	input_param.pn_nconf = 100;
	input_param.E_0 = 200;
	input_param.pn_dim = cudaEM::FP_Dim(true, true, true);
	/*******************Atom settings************************/
	direct_xyzreader(input_param.input_atom_coordinates);
	input_param.is_crystal = true;
	input_param.slice_interval = 0.5;
	input_param.specimen_ratation = false;
	input_param.blank_ratio = 0.1;
	/**********************Grid settings**************************/
	input_param.img_col = col;
	input_param.img_row = row;
	input_param.GPU_num = 1;
	/****************initialize the input params******************/
	input_param.obj_lens.c_10 = -81.0;
	input_param.obj_lens.outer_aper_ang = 0.080;
	input_param.obj_lens.dsf_sigma = 10.0;
	input_param.obj_lens.dsf_npoints = 30;
	/************************************************************/
	input_param.validate_parameters();
	input_param.initial_OL();
	cudaEM::TEM_simulator<float> simulation;
	cudaEM::output_Multi_Data<float> output_data;
	simulation.set_input_data(&input_param, &output_data);
	std::cout << "finish initialization" << std::endl;
	simulation.TEM_multiGPU();
}
void STEM()
{
	cudaEM::multislice_params<float> input_param;
	input_param.simulation_type = cudaEM::eTEMST_STEM;
	input_param.potential_type = cudaEM::ePT_Lobato_0_12;
	input_param.phonon_model = cudaEM::ePM_Frozen_Phonon;
	input_param.enable_fine_slicing = true;
	input_param.illumination_model = cudaEM::eIM_Trans_Cross_Coef;
	input_param.temporal_spatial_incoh = cudaEM::eTSI_Temporal;
	/******************Phonon settings**********************/
	input_param.pn_nconf = 1;
	input_param.E_0 = 200;
	input_param.pn_dim = cudaEM::FP_Dim(true, true, true);
	/*******************Atom settings************************/
	phonon_builder(input_param.input_atom_coordinates);
	input_param.is_crystal = false;
	input_param.slice_interval = 1.0;
	input_param.specimen_ratation = false;
	/**********************Grid settings**************************/
	input_param.GPU_num = 1;
	/****************initialize the input params******************/
	input_param.cond_lens.c_10 = 0.0f;
	input_param.cond_lens.c_30 = 0.0f;
	input_param.cond_lens.inner_aper_ang = 0.0f;
	input_param.cond_lens.outer_aper_ang = 0.03f;

	input_param.detector.resize(3);
	input_param.detector.g_inner[0] = 15;
	input_param.detector.g_outer[0] = 30;
	input_param.detector.g_inner[1] = 30;
	input_param.detector.g_outer[1] = 70;
	input_param.detector.g_inner[2] = 70;
	input_param.detector.g_outer[2] = 250;
	input_param.validate_parameters();
	input_param.initial_CL();
	std::vector<float> scan_range(4);
	scan_range[0] = 0.0f;
	scan_range[1] = 0.0f;
	scan_range[2] = 1.0f;
	scan_range[3] = 1.0f;
	input_param.scanning_step = 2;
	input_param.scanning_ratio = 4;
	input_param.set_Scanning(scan_range);
	/************************************************************/
	cudaEM::scan_simulator<float> simulation;
	cudaEM::output_Multi_Data<float> output_data;
	simulation.set_input_data(&input_param, &output_data);
	simulation.STEM_multiGPU();
	/*************************************************************/
	int row = output_data.row;
	int col = output_data.col;
	int pixel_num = row*col;
	std::vector<float> HAADF_data(pixel_num);
	std::vector<float> LAADF_data(pixel_num);
	std::vector<float> ABF_data(pixel_num);
	memcpy(LAADF_data.data(),
		&output_data.STEM_intensity[row*col],
		pixel_num * sizeof(float));
	memcpy(HAADF_data.data(),
		&output_data.STEM_intensity[2 * row*col],
		pixel_num * sizeof(float));
	memcpy(ABF_data.data(),
		&output_data.STEM_intensity[0],
		pixel_num * sizeof(float));
	cudaEM::EMdata_IO rawSaver("D:/goldSTEM.h5", "gold");
	rawSaver.create_file();
	rawSaver.write_STEM_data(row, col,HAADF_data, LAADF_data, ABF_data);
}
int main()
{
	STEM();
	system("pause");
	return 0;
}