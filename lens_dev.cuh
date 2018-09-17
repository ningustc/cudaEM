#ifndef LENS_DEV_H
#define LENS_DEV_H
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_basic_def.cuh"
#include "misc_dev.cuh"
namespace cudaEM
{
	template <class T>
	struct Lens_Dev
	{
		int m; 					// Momentum of the vortex
		T c_10; 				// defocus, angstrom
		T c_30; 				// Third order spherical aberration in angstrom.
		T phi_12; 				// Azimuthal angle of 2-fold astigmatism (rad)
		T phi_21; 				// Azimuthal angle of axial coma (rad)
		T phi_23; 				// Azimuthal angle of 3-fold astigmatism (rad)
		T phi_32; 				// Azimuthal angle of axial star aberration (rad)
		T phi_34; 				// Azimuthal angle of 4-fold astigmatism (rad)
		T phi_41; 				// Azimuthal angle of 4th order axial coma (rad)
		T phi_43; 				// Azimuthal angle of 3-lobe aberration (rad)
		T phi_45; 				// Azimuthal angle of 5-fold astigmatism (rad)
		T phi_52; 				// Azimuthal angle of 5th order axial star aberration (rad)
		T phi_54; 				// Azimuthal angle of 5th order rosette aberration (rad)
		T phi_56; 				// Azimuthal angle of 6-fold astigmatism (rad)

		T dsf_sigma; 			// Standard deviation of the defocus spread function(Å)
		int dsf_npoints; 		// Number of integration points of the defocus spread function
		T dsf_iehwgd; 			// e^-1 half-width value of the Gaussian distribution

		T ssf_sigma; 			// Standard deviation of the source spread function
		int ssf_npoints; 		// Number of integration points of the source spread function
		T ssf_iehwgd; 			// e^-1 half-width value of the Gaussian distribution
		T ssf_beta; 			// divergence semi-angle (rad)
		T gamma; 				// Relativistic factor
		T lambda; 				// wavelength(Angstrom)
		T lambda_squared; 		// wavelength(Angstrom)


		T c_c_10; 				// -pi*c_10*lambda
		T c_c_12; 				// -pi*c_12*lambda

		T c_c_21; 				// -2*pi*c_21*lambda^2/3
		T c_c_23; 				// -2*pi*c_23*lambda^2/3

		T c_c_30; 				// -pi*c_30*lambda^3/2
		T c_c_32; 				// -pi*c_32*lambda^3/2
		T c_c_34; 				// -pi*c_34*lambda^3/2

		T c_c_41; 				// -2*pi*c_41*lambda^4/5
		T c_c_43; 				// -2*pi*c_43*lambda^4/5
		T c_c_45; 				// -2*pi*c_45*lambda^4/5

		T c_c_50; 				// -pi*c_50*lambda^5/3
		T c_c_52; 				// -pi*c_52*lambda^5/3
		T c_c_54; 				// -pi*c_54*lambda^5/3
		T c_c_56; 				// -pi*c_56*lambda^5/3

		T g2_min; 				// inner_aper_ang/lambda
		T g2_max; 				// outer_aper_ang/lambda
		int ngxs; 				// Number of source sampling points x
		int ngys; 				// Number of source sampling points y
		T dgxs; 				// source sampling m_size;
		T dgys; 				// source sampling m_size;
		T g2_maxs; 				// q maximum square;

		Lens_Dev() : m(0), c_10(0), c_30(0), phi_12(0), phi_21(0), phi_23(0), phi_32(0), phi_34(0), phi_41(0),
			phi_43(0), phi_45(0), phi_52(0), phi_54(0), phi_56(0), dsf_sigma(0), dsf_npoints(0), dsf_iehwgd(0),
			ssf_sigma(0), ssf_npoints(0), ssf_iehwgd(0), ssf_beta(0), gamma(0), lambda(0), lambda_squared(0),
			g2_min(0), g2_max(0), ngxs(0), ngys(0), dgxs(0), dgys(0), g2_maxs(0),
			c_c_10(0), c_c_12(0), c_c_21(0), c_c_23(0), c_c_30(0), c_c_32(0), c_c_34(0), c_c_41(0),
			c_c_43(0), c_c_45(0), c_c_50(0), c_c_52(0), c_c_54(0), c_c_56(0) {}

		template <class HLens>
		void assign(HLens &lens)
		{
			m = lens.m;
			c_10 = lens.c_10;
			c_30 = lens.c_30;
			phi_12 = lens.phi_12;
			phi_21 = lens.phi_21;
			phi_23 = lens.phi_23;
			phi_32 = lens.phi_32;
			phi_34 = lens.phi_34;
			phi_41 = lens.phi_41;
			phi_43 = lens.phi_43;
			phi_45 = lens.phi_45;
			phi_52 = lens.phi_52;
			phi_54 = lens.phi_54;
			phi_56 = lens.phi_56;
			dsf_sigma = lens.dsf_sigma;
			dsf_npoints = lens.dsf_npoints;
			ssf_beta = lens.ssf_beta;
			ssf_npoints = lens.ssf_npoints;

			gamma = lens.gamma;
			lambda = lens.lambda;
			lambda_squared = std::pow(lambda, 2);

			//export the private variables to device lens.
			c_c_10 = lens.export_c_c_10();
			c_c_12 = lens.export_c_c_12();

			c_c_21 = lens.export_c_c_21();
			c_c_23 = lens.export_c_c_23();

			c_c_30 = lens.export_c_c_30();
			c_c_32 = lens.export_c_c_32();
			c_c_34 = lens.export_c_c_34();

			c_c_41 = lens.export_c_c_41();
			c_c_43 = lens.export_c_c_43();
			c_c_45 = lens.export_c_c_45();

			c_c_50 = lens.export_c_c_50();
			c_c_52 = lens.export_c_c_52();
			c_c_54 = lens.export_c_c_54();
			c_c_56 = lens.export_c_c_56();

			dsf_iehwgd = lens.export_df_hw();
			ssf_iehwgd = lens.export_sf_hw();
			ssf_sigma = lens.export_sf_sigma();
			g2_min = lens.export_g2_min();
			g2_max = lens.export_g2_max();

			ngxs = lens.export_gx_number();
			ngys = lens.export_gy_number();
			dgxs = lens.export_dgxs();
			dgys = lens.export_dgys();
			g2_maxs = lens.export_g2_maxs();
		}

		template <class HLens>
		void operator=(HLens &lens)
		{
			assign(lens);
		}
		void set_dsf_sigma(T dsf_sigma_i)
		{
			dsf_sigma = dsf_sigma_i;
			dsf_iehwgd = c_2i2*dsf_sigma;
		}
		void set_ssf_sigma(T ssf_beta_i)
		{
			ssf_beta = ssf_beta_i;
			ssf_iehwgd = std::sin(ssf_beta) / lambda;
			ssf_sigma = ssf_iehwgd / c_2i2;
		}
		inline T gxs(const int &col_idx) const { return static_cast<T>(col_idx)*dgxs; }

		inline T gys(const int &row_idx) const { return static_cast<T>(row_idx)*dgys; }

		inline T g2s(const int &col_idx, const int &row_idx) const
		{
			T gxi = gxs(col_idx);
			T gyi = gys(row_idx);
			return gxi*gxi + gyi*gyi;
		}
		inline void set_defocus(T f_i)
		{
			c_10 = f_i;
			c_c_10 = (isZero(c_10)) ? 0 : -c_Pi*c_10*lambda;
		}
		/**********************function running on CUDA kernels****************************/
		__device__ bool is_phi_required() const
		{
			auto bb = nonZero_dev(m) || nonZero_dev(c_c_12) || nonZero_dev(c_c_21) || nonZero_dev(c_c_23) || nonZero_dev(c_c_32) || nonZero_dev(c_c_34);
			bb = bb || nonZero_dev(c_c_41) || nonZero_dev(c_c_43) || nonZero_dev(c_c_45) || nonZero_dev(c_c_52) || nonZero_dev(c_c_54) || nonZero_dev(c_c_56);
			return bb;
		}
		__device__ T eval_m(const T &phi) const
		{
			return (nonZero_dev(phi)) ? (m*phi) : 0;
		}
		/************************************************************************/
		__device__ T eval_c_10(const T &g2) const
		{
			return (nonZero_dev(c_c_10)) ? (c_c_10*g2) : 0;
		}
		__device__ T eval_c_12(const T &g2, const T &phi) const
		{
			return (nonZero_dev(c_c_12)) ? (c_c_12*g2*sin(2 * (phi - phi_12))) : 0;
		}
		/************************************************************************/
		__device__ T eval_c_21(const T &g3, const T &phi) const
		{
			return (nonZero_dev(c_c_21)) ? (c_c_21*g3*sin(phi - phi_21)) : 0;
		}

		__device__ T eval_c_23(const T &g3, const T &phi) const
		{
			return (nonZero_dev(c_c_23)) ? (c_c_23*g3*sin(3 * (phi - phi_23))) : 0;
		}
		__device__ T eval_c_21_c_23(const T &g3, const T &phi) const
		{
			return (eval_c_21(g3, phi) + eval_c_23(g3, phi));
		}
		/************************************************************************/
		__device__ T eval_c_30(const T &g4) const
		{
			return (nonZero_dev(c_c_30)) ? (c_c_30*g4) : 0;
		}
		__device__ T eval_c_32(const T &g4, const T &phi) const
		{
			return (nonZero_dev(c_c_32)) ? (c_c_32*g4*sin(2 * (phi - phi_32))) : 0;
		}
		__device__ T eval_c_34(const T &g4, const T &phi) const
		{
			return (nonZero_dev(c_c_34)) ? (c_c_34*g4*sin(4 * (phi - phi_34))) : 0;
		}
		__device__ T eval_c_32_c_34(const T &g4, const T &phi) const
		{
			return (eval_c_32(g4, phi) + eval_c_34(g4, phi));
		}
		/************************************************************************/
		__device__ T eval_c_41(const T &g5, const T &phi) const
		{
			return (nonZero_dev(c_c_41)) ? (c_c_41*g5*sin(phi - phi_41)) : 0;
		}
		__device__ T eval_c_43(const T &g5, const T &phi) const
		{
			return (nonZero_dev(c_c_43)) ? (c_c_43*g5*sin(3 * (phi - phi_43))) : 0;
		}
		__device__ T eval_c_45(const T &g5, const T &phi) const
		{
			return (nonZero_dev(c_c_45)) ? (c_c_45*g5*sin(5 * (phi - phi_45))) : 0;
		}
		__device__ T eval_c_41_c_43_c_45(const T &g5, const T &phi) const
		{
			return (eval_c_41(g5, phi) + eval_c_43(g5, phi) + eval_c_45(g5, phi));
		}
		/************************************************************************/
		__device__ T eval_c_50(const T &g6) const
		{
			return (nonZero_dev(c_c_50)) ? (c_c_50*g6) : 0;
		}
		__device__ T eval_c_52(const T &g6, const T &phi) const
		{
			return (nonZero_dev(c_c_52)) ? (c_c_52*g6*sin(2 * (phi - phi_52))) : 0;
		}
		__device__ T eval_c_54(const T &g6, const T &phi) const
		{
			return (nonZero_dev(c_c_54)) ? (c_c_54*g6*sin(4 * (phi - phi_54))) : 0;
		}
		__device__ T eval_c_56(const T &g6, const T &phi) const
		{
			return (nonZero_dev(c_c_56)) ? (c_c_56*g6*sin(6 * (phi - phi_56))) : 0;
		}
		__device__ T eval_c_52_c_54_c_56(const T &g6, const T &phi) const
		{
			return (eval_c_52(g6, phi) + eval_c_54(g6, phi) + eval_c_56(g6, phi));
		}
	};
}
#endif