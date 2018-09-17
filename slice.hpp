#pragma once
#ifndef SLICE_H
#define SLICE_H
#include <cstdlib>
namespace cudaEM
{
	template <class T>
	struct Slice
	{
		Slice() : z_start(0), z_end(0), z_int_start(0),
			z_int_end(0), iatom_start(1), iatom_end(0) {}

		T z_start; 			     // Initial z-position
		T z_end; 			     // Final z-position
		T z_int_start; 		     // Initial z-position
		T z_int_end; 		     // Final z-position
		std::size_t iatom_start; // Index to initial z-position
		std::size_t iatom_end; 	 // Index to final z-position
		T dz() const
		{
			return std::fabs(z_end - z_start);
		}
		T z_m() const
		{
			return 0.5*(z_end + z_start);
		}
	};
}
#endif