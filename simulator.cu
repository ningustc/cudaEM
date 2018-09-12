#pragma once
#include "simulator.h"
#include "scan_simulator.cuh"
#include "multislice_params_dev.cuh"
#include "single_simulator.cuh"
#include "single_simulator.cuh"
using namespace cudaEM;
void mulSimulator::executionTEM(const cudaEM::multislice_params<float>& input_params,
	cudaEM::output_Multi_Data<float> *output_data_ptr)
{
	cudaEM::multislice_params<float> input_param;
	input_param.assign(input_params);
	/****************initialize the input params******************/
	input_param.validate_parameters();
	input_param.initial_OL();
	cudaEM::TEM_simulator<float> simulation;
	simulation.set_input_data(&input_param, output_data_ptr);
	simulation.TEM_multiGPU();
}
void mulSimulator::executionCBED(const cudaEM::multislice_params<float>& input_params,
	cudaEM::output_Multi_Data<float> *output_data_ptr)
{
	cudaEM::multislice_params<float> input_param;
	input_param.assign(input_params);
	input_param.validate_parameters();
	input_param.initial_CL();
	cudaEM::TEM_simulator<float> simulation;
	simulation.set_input_data(&input_param, output_data_ptr);
	simulation.TEM_multiGPU();
}
void mulSimulator::executionSTEM(const cudaEM::multislice_params<float>& input_params,
	cudaEM::output_Multi_Data<float> *output_data)
{
	cudaEM::multislice_params<float> input_param;
	input_param.assign(input_params);
	/******************Phonon settings**********************/
	input_param.is_crystal = false;
	/****************initialize the input params******************/
	input_param.validate_parameters();
	input_param.initial_CL();
	std::vector<float> scan_range(4);
	scan_range[0] = 0.0f;
	scan_range[1] = 0.0f;
	scan_range[2] = 1.0f;
	scan_range[3] = 1.0f;
	input_param.set_Scanning(scan_range);
	cudaEM::scan_simulator<float> simulation;
	simulation.set_input_data(&input_param, output_data);
	simulation.STEM_multiGPU();
}