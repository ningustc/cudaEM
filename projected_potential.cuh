#include "host_basic_def.hpp"
#include "specimen.cuh"
#include "basic_functions.hpp"
#include "multislice_params_dev.cuh"
#include "device_kernels.cuh"
#include "device_containers.cuh"
namespace cudaEM
{
	template <class T>
	class Projected_Potential : public Specimen<T> 
	{
	public:
		Projected_Potential() :Specimen<T>(){}
		~Projected_Potential()
		{
			cudaFree(atom_Vp_dev_ptr);
		}
		void set_input_data(multislice_params_dev<T> *input_multislice_i)
		{
			Specimen<T>::set_input_data(input_multislice_i);
			multisliceparam_ptr = input_multislice_i;
			grid_dev_ptr = &input_multislice_i->grid_dev;
			qz_dev_ptr = input_multislice_i->qz_dev;
			element_type_dev_ptr = &input_multislice_i->element_types_dev;
			pixel_num = grid_dev_ptr->Grid_size();
			/******resize and allocate space for device and host parameters***********/
			atom_Vp_host.resize(c_Atom_Batch);
			/***********get the pointer of the stored device data*****************/
			cudaMalloc((void **)&atom_Vp_dev_ptr, 
				c_Atom_Batch*sizeof(Combined_Element_Coeff_Ptrs<T>));
			Blk_Dim_2D = input_multislice_i->Blk_Dim_2D;
			Grd_Dim_2D = input_multislice_i->Grd_Dim_2D;
			Blk_Dim_1D = dim3(c_nqz);
			if (input_multislice_i->enable_fine_slicing)
			{
				atom_cubic_dev.resize(c_nR*c_Atom_Batch);
				for (auto istream = 0; istream < c_Atom_Batch; istream++)
				{
					/***********Assign the address of the to be determined variables****************************/
					atom_Vp_host[istream].c0_dev = atom_cubic_dev.c0_dev + istream*c_nR;
					atom_Vp_host[istream].c1_dev = atom_cubic_dev.c1_dev + istream*c_nR;
					atom_Vp_host[istream].c2_dev = atom_cubic_dev.c2_dev + istream*c_nR;
					atom_Vp_host[istream].c3_dev = atom_cubic_dev.c3_dev + istream*c_nR;
				}
			}
		}
		void generate_potential_by_slice(const int &islice_0, const int &islice_e, T* V_dev_ptr, T coeff = T(1.0))
		{
			cudaMemset(V_dev_ptr, 0, sizeof(T)*pixel_num);
			if ((islice_0 < 0) || (islice_e >= this->slicing.slices.size()))
			{
				return;
			}
			std::size_t start_index = this->slicing.slices[islice_0].iatom_start;
			std::size_t end_index = this->slicing.slices[islice_e].iatom_end;
			if (start_index>end_index)
			{
				return;
			}
			generate_potential_by_height(this->slicing.slices[islice_0].z_start, this->slicing.slices[islice_e].z_end,
				start_index, end_index, V_dev_ptr, coeff);
		}
		void generate_potential_single_slice(const int &islice, T* V_dev_ptr, T coeff=T(1.0))
		{
			generate_potential_by_slice(islice, islice, V_dev_ptr, coeff);
		}
	private:
		//warning: the input potential should be initialized to zero.
		void generate_potential_by_height(const T &z_0, const T &z_e, const int &iatom_0,
			const int &iatom_e, T* V_dev_ptr, T coeff = T(1.0))
		{
			T z_interval = z_e - z_0;
			/**************iterate along the atom numbers***************/
			int iatoms = iatom_0;
			while (iatoms <= iatom_e)
			{
				//the atom numbers, and i_atom is the starting index
				int n_atoms = std::min(c_Atom_Batch, iatom_e - iatoms + 1);
				//set the pointer
				set_atom_Vp_host(z_0, z_e, iatoms, n_atoms);
				//if set the finner slicing
				if (multisliceparam_ptr->enable_fine_slicing)
				{
					Grd_Dim_1D = dim3(c_nR, n_atoms);
					gen_sa_coeff_kernel<T> << <Grd_Dim_1D, Blk_Dim_1D >> > (multisliceparam_ptr->potential_type,
						qz_dev_ptr, atom_Vp_dev_ptr, z_interval);
					fill_spl_coeff_kernel<T> << <dim3(n_atoms), dim3(c_nR) >> > (atom_Vp_dev_ptr);
				}
				gen_slice_potential_kernel<T> << <Grd_Dim_2D, Blk_Dim_2D >> > (*grid_dev_ptr, atom_Vp_dev_ptr, V_dev_ptr, n_atoms, coeff);
				iatoms += n_atoms;
			}
		}
		void set_atom_Vp_host(const T &z_0, const T &z_e, int iatoms, int n_atoms)
		{
			for (auto istream = 0; istream < n_atoms; istream++)
			{
				//require the data from the specimen class
				auto iZ = this->atoms.Z[iatoms];
				auto Z_index = this->atoms.get_Z_index(iZ);
				atom_Vp_host[istream].x = this->atoms.x[iatoms];
				atom_Vp_host[istream].y = this->atoms.y[iatoms];
				atom_Vp_host[istream].occ = this->atoms.occ[iatoms];
				atom_Vp_host[istream].R_min_squared = (*element_type_dev_ptr)[Z_index].R2_min();
				atom_Vp_host[istream].R_tap_squared = (*element_type_dev_ptr)[Z_index].R2_tap();
				atom_Vp_host[istream].tap_cf = (*element_type_dev_ptr)[Z_index].tap_cf;
				atom_Vp_host[istream].R2_dev = (*element_type_dev_ptr)[Z_index].R2_Cubic_dev_ptr;

				/********************compute the integration range of this atom*************************/
				T maximum_radius = (*element_type_dev_ptr)[Z_index].R_max;
				atom_Vp_host[istream].R_max_squared = std::pow(maximum_radius, 2);
				
				if (multisliceparam_ptr->enable_fine_slicing)
				{
					/******************Assign the height information**********************/
					atom_Vp_host[istream].z_middle = 0.5*(z_e + z_0) - this->atoms.z[iatoms];
					/**********compute the Vr linear and non-linear variables*************/
					atom_Vp_host[istream].cl_dev = (*element_type_dev_ptr)[Z_index].Vr_coeff_dev.cl_dev;
					atom_Vp_host[istream].cnl_dev = (*element_type_dev_ptr)[Z_index].Vr_coeff_dev.cnl_dev;
				}
				else
				{
					/********************For direct projected potential without fine slicing******************/
					atom_Vp_host[istream].c0_dev = (*element_type_dev_ptr)[Z_index].VR_Cubic_coeff_dev.c0_dev;
					atom_Vp_host[istream].c1_dev = (*element_type_dev_ptr)[Z_index].VR_Cubic_coeff_dev.c1_dev;
					atom_Vp_host[istream].c2_dev = (*element_type_dev_ptr)[Z_index].VR_Cubic_coeff_dev.c2_dev;
					atom_Vp_host[istream].c3_dev = (*element_type_dev_ptr)[Z_index].VR_Cubic_coeff_dev.c3_dev;
				}
				iatoms++;
			}
			//copy the content from host to device
			cudaMemcpy(atom_Vp_dev_ptr, atom_Vp_host.data(), 
				sizeof(Combined_Element_Coeff_Ptrs<T>)*n_atoms, cudaMemcpyHostToDevice);
		}
	private:
		std::size_t pixel_num;
		multislice_params_dev<T> *multisliceparam_ptr;
		Grid_2d_Dev<T> *grid_dev_ptr;
		T *qz_dev_ptr;
		std::vector<Element_Coeffs_Dev<T>> *element_type_dev_ptr;
		/********************pointers combo structure*********************/
		std::vector<Combined_Element_Coeff_Ptrs<T>> atom_Vp_host;
		Combined_Element_Coeff_Ptrs<T> *atom_Vp_dev_ptr;
		dim3 Blk_Dim_2D, Grd_Dim_2D;
		dim3 Blk_Dim_1D, Grd_Dim_1D;
		/***************single atom property determiend by height*********/
		Combined_CSI_Data_Dev<T> atom_cubic_dev;
	};
}