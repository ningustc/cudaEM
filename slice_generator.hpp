#pragma once
#ifndef SLICING_H
#define SLICING_H
#include <algorithm>
#include <cmath>
#include "slice.hpp"
#include "host_basic_def.hpp"
#include "atom_coor_data.hpp"
namespace cudaEM
{
	template <class T>
	struct Slice_Generator
	{
	public:
		Slice_Generator() : m_atoms_ptr(nullptr),slice_border_type(eMB_Middle)
		{
		}
		/************************************************************************/
		/* atoms_r is the static atom, and atoms is the vibrated or rotate atoms*/
		/************************************************************************/
		void set_input_data(bool pot_sli, T interval,
			Atom_Coord_Data<T> *atoms_ptr_i, eMatch_Border border_type= eMB_Middle)
		{
			Slice_interval = interval;
			fine_slicing = pot_sli;
			m_atoms_ptr = atoms_ptr_i;
			slice_border_type = border_type;
		}
		T slice_interval(int islice_0, int islice_e)
		{
			return (islice_e < slices.size()) ? (slices[islice_e].z_end - slices[islice_0].z_start) : 0.0;
		}

		T single_slice_interval(int islice)
		{
			return slice_interval(islice, islice);
		}

		T slice_middle_z(int islice)
		{
			return (islice < slices.size()) ? slices[islice].z_m() : 0.0;
		}

		T slice_middle_interval(int islice_0, int islice_e)
		{
			return std::fabs(slice_middle_z(islice_e) - slice_middle_z(islice_0));
		}
		void calculate()
		{
			std::vector<T>  slice_bounds = get_slice_boundary(*m_atoms_ptr);
			get_slicing(slice_bounds, *m_atoms_ptr);
		}
		std::vector<Slice<T>> slices;
	private:
		// Identify planes: Require that the atoms to be sorted along z
		std::vector<T> get_slice_boundary(const Atom_Coord_Data<T> &atoms)
		{
			std::vector<T> z_bound_planes;
			if (atoms.size() == 0)
			{
				return z_bound_planes;
			}
			if (fine_slicing)
			{
				z_bound_planes = gene_bounds(atoms.z_int_min, atoms.z_int_max);
			}
			else
			{
				std::vector<T> z_val;
				z_val.reserve(atoms.size());
				for (auto iz = 0; iz < atoms.size(); iz++)
				{
					z_val.push_back(atoms.z[iz]);
				}
				std::sort(z_val.begin(), z_val.end());
				T z_val_min = z_val.front(); T z_val_max = z_val.back();
				z_bound_planes = gene_bounds(z_val_min, z_val_max);
			}
			return z_bound_planes;
		}
		void get_slicing(const std::vector<T> &z_slice,const Atom_Coord_Data<T> &atoms)
		{
			std::size_t slice_size= z_slice.size() - 1;
			slices.resize(slice_size);
			for (auto islice = 0; islice < slices.size(); islice++)
			{
				bool Inc_Borders = false;
				slices[islice].z_start = z_slice[islice];
				slices[islice].z_end = z_slice[islice + 1];
				if (!fine_slicing)
				{
					slices[islice].z_int_start = slices[islice].z_start;
					slices[islice].z_int_end = slices[islice].z_end;
					Inc_Borders = false;
				}
				else
				{
					T z_m = slices[islice].z_m();

					slices[islice].z_int_start = std::fmin(z_m - atoms.R_int_max, slices[islice].z_start);
					slices[islice].z_int_end = std::fmax(z_m + atoms.R_int_max, slices[islice].z_end);
					Inc_Borders = true;
				}
				fd_by_z(atoms.z, slices[islice].z_int_start, slices[islice].z_int_end, slices[islice].iatom_start, slices[islice].iatom_end, Inc_Borders);
			}
			slices.shrink_to_fit();
		}
		// find atoms in slices
		void fd_by_z(const std::vector<T> &z, T z_start, T z_end, std::size_t &idx_z_start, std::size_t &idx_z_end, bool Inc_Borders)
		{
			z_start = (Inc_Borders) ? (z_start - z_eps) : z_start;
			z_end = (Inc_Borders) ? (z_end + z_eps) : z_end;

			if ((z_start > z_end) || (z.back() < z_start) || (z_end < z.front()))
			{
				idx_z_start = 1;
				idx_z_end = 0;
				return;
			}

			idx_z_start = (z_start <= z.front()) ? 0 : (std::lower_bound(z.begin(), z.end(), z_start) - z.begin());
			idx_z_end = (z.back() <= z_end) ? (z.size() - 1) : (std::lower_bound(z.begin(), z.end(), z_end) - z.begin() - 1);

			if ((idx_z_start > idx_z_end) || (z[idx_z_end] < z_start) || (z_end < z[idx_z_start]))
			{
				idx_z_start = 1;
				idx_z_end = 0;
			}
		}
		std::vector<T> gene_bounds(T v_min, T v_max)
		{
			std::vector<T> slice_bounds;
			T s_v = v_max - v_min;
			const int nv = static_cast<int>(std::floor((s_v+ z_eps) / Slice_interval))+2;
			//get the total thickness. 
			slice_bounds.resize(nv);
			switch (slice_border_type)
			{
			case eMB_Min:
			{
				T z_bottom = v_min- 2 * z_eps;
				for (auto islice = 0; islice < nv; islice++)
				{
					slice_bounds[islice] = z_bottom + islice*Slice_interval;
				}
			}
			break;
			case eMB_Max:
			{
				T z_top = v_max + 2 * z_eps;
				for (auto islice = 0; islice < nv; islice++)
				{
					slice_bounds[nv - 1 - islice] = z_top - islice*Slice_interval;
				}
			}
			break;
			case eMB_Middle:
			{
				T z_bottom = T(0.5)*(v_min+v_max - (nv-1)*Slice_interval);
				for (auto islice = 0; islice < nv; islice++)
				{
					slice_bounds[islice] = z_bottom + islice*Slice_interval;
				}
			}
			break;
			}
			return slice_bounds;
		}
		/***********************************/
		const T z_eps= epsilon_rel<T>();
		bool fine_slicing;
		eMatch_Border slice_border_type;
		Atom_Coord_Data<T> *m_atoms_ptr;
		T Slice_interval;
	};

}
#endif
