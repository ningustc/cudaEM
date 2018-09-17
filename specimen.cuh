#pragma once
#ifndef SPECIMEN_H
#define SPECIMEN_H
#include "host_basic_def.hpp"
#include "atom_coor_data.hpp"
#include "multislice_params_dev.cuh"
#include "slice_generator.hpp"
#include "random_lib.hpp"
namespace cudaEM
{
	template <class T>
	class Specimen
	{
	public:
		Specimen() : multisliceparam_ptr(nullptr) {}
		void set_input_data(multislice_params_dev<T> *input_multislice_i)
		{
			multisliceparam_ptr = input_multislice_i;
			atoms_u = &multisliceparam_ptr->input_atoms;
			atoms.assign(*atoms_u);
			R_max_ptr = &multisliceparam_ptr->R_maximums;
			grid_dev_ptr = &multisliceparam_ptr->grid_dev;

			if (multisliceparam_ptr->simulation_type== eTEMST_HRTEM)
			{
				border_type = eMB_Max;
			}
			else
			{
				border_type = eMB_Min;
			}
			/*****************initialize the slicing and compute the slices***************************/
			slicing.set_input_data(multisliceparam_ptr->enable_fine_slicing, 
				multisliceparam_ptr->grid_dev.dz, &atoms, border_type);
			slicing.calculate();
		}
		void update_atom(Atom_Coord_Data<T> *atoms_i)
		{
			atoms_u = atoms_i;
			atoms.assign(*atoms_u);
			atoms.get_statistic(R_max_ptr);
			slicing.set_input_data(multisliceparam_ptr->enable_fine_slicing,
				grid_dev_ptr->dz, &atoms, border_type);
			slicing.calculate();
		}
		//Theta is the tilting angle along the perpendicular direction, and alpha is the other angle.
		T tilted_single_slice_interval(const int &islice)
		{
			return slicing.single_slice_interval(islice) / std::cos(multisliceparam_ptr->theta);
		}
		T tilted_slice_middle_interval(const int &islice_0, const int &islice_e)
		{
			return slicing.slice_middle_interval(islice_0, islice_e) / std::cos(multisliceparam_ptr->theta);
		}
		multislice_params_dev<T> *multisliceparam_ptr;
		Atom_Coord_Data<T> atoms; 							// displaced atoms
		Slice_Generator<T> slicing; 						// storing the generated slices.
	private:
		Atom_Coord_Data<T> *atoms_u;
		std::vector<T> *   R_max_ptr;
		Grid_2d_Dev<T>* grid_dev_ptr;
		eMatch_Border border_type;
	};
}
#endif