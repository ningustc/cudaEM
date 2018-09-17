#ifndef LENS_MODULATION_H
#define LENS_MODULATION_H
#include "host_basic_def.hpp"
#include "device_basic_def.cuh"
#include "device_kernels.cuh"
#include "multislice_params_dev.cuh"
namespace cudaEM
{
	template <class T>
	class Lens_Modulation
	{
	public:
		Lens_Modulation() : multisliceparam_ptr(nullptr) {}
		~Lens_Modulation()
		{
			cudaFree(exit_wave_ptr);
		}
		void set_input_data(multislice_params_dev<T> *input_multislice_i)
		{
			multisliceparam_ptr = input_multislice_i;
			fft_2d_ptr = &input_multislice_i->fft_trans_dev;
			grid_dev_ptr = &input_multislice_i->grid_dev;
			OL_dev_ptr = &input_multislice_i->obj_lens_dev;
			pixel_num = grid_dev_ptr->Grid_size();
			Blk_Dim_2D = input_multislice_i->Blk_Dim_2D;
			Grd_Dim_2D = input_multislice_i->Grd_Dim_2D;
			Blk_Dim_1D = input_multislice_i->Blk_Dim_1D;
			Grd_Dim_1D = input_multislice_i->Grd_Dim_1D;
			cudaMalloc((void**)&exit_wave_ptr, sizeof(T_complex<T>)*pixel_num);
			/*********************************************************/
			qt_ptr = &input_multislice_i->temporal_quard;
			qs_ptr = &input_multislice_i->spatial_quard;
		}
		void operator()(T_complex<T>* wave_ptr, T* intensity_ptr)
		{
			switch (multisliceparam_ptr->illumination_model)
			{
			case eIM_Coherent:
			{
				CTF_TEM(wave_ptr, intensity_ptr);
			}
			break;
			case eIM_Partial_Coherent:
			{
				PCTF_TEM(multisliceparam_ptr->temp_spat_incoh, wave_ptr, intensity_ptr);
			}
			break;
			case eIM_Trans_Cross_Coef:
			{
				TCC_TEM(multisliceparam_ptr->temp_spat_incoh, wave_ptr, intensity_ptr);
			}
			break;
			}
		}
		void operator()(T_complex<T>* wave_ptr, T* real_ptr, T* imag_ptr)
		{
			switch (multisliceparam_ptr->illumination_model)
			{
			case eIM_Coherent:
			{
				CTF_TEM(wave_ptr, real_ptr, imag_ptr);
			}
			break;
			case eIM_Partial_Coherent:
			{
				PCTF_TEM(multisliceparam_ptr->temp_spat_incoh, wave_ptr, real_ptr, imag_ptr);
			}
			break;
			case eIM_Trans_Cross_Coef:
			{
				TCC_TEM(multisliceparam_ptr->temp_spat_incoh, wave_ptr, real_ptr, imag_ptr);
			}
			break;
			}
		}
	private:
		/******************************************************************/
		void CTF_TEM(T_complex<T>* wave_ptr, T* intensity_ptr)
		{
			apply_CTF_kernel<T> << <Grd_Dim_2D, Blk_Dim_2D >> > (*grid_dev_ptr, *OL_dev_ptr, 0, 0, wave_ptr, exit_wave_ptr);
			fft_2d_ptr->inverse(exit_wave_ptr);
			wave_amplitude<T> << <Grd_Dim_1D, Blk_Dim_1D >> > (exit_wave_ptr, intensity_ptr, pixel_num);
		}
		void CTF_TEM(T_complex<T>* wave_ptr, T* real_ptr, T* imag_ptr)
		{
			apply_CTF_kernel<T> << <Grd_Dim_2D, Blk_Dim_2D >> > (*grid_dev_ptr, *OL_dev_ptr, 0, 0, wave_ptr, exit_wave_ptr);
			fft_2d_ptr->inverse(exit_wave_ptr);
			wave_spliter<T> << <Grd_Dim_1D, Blk_Dim_1D >> > (exit_wave_ptr, real_ptr, imag_ptr, pixel_num);
		}
		/******************************************************************/
		void PCTF_TEM(const eTemporal_Spatial_Incoh &temporal_spatial_incoh, T_complex<T>* wave_ptr, T* intensity_ptr)
		{
			T dsf_sigma = OL_dev_ptr->dsf_sigma;
			T ssf_beta = OL_dev_ptr->ssf_beta;
			if (temporal_spatial_incoh == eTSI_Temporal)
			{
				OL_dev_ptr->set_ssf_sigma(0);
			}
			if (temporal_spatial_incoh == eTSI_Spatial)
			{
				OL_dev_ptr->set_dsf_sigma(0);
			}
			apply_PCTF_kernel<T> << <Grd_Dim_2D, Blk_Dim_2D >> > (*grid_dev_ptr, *OL_dev_ptr, wave_ptr, exit_wave_ptr);
			fft_2d_ptr->inverse(exit_wave_ptr);
			wave_amplitude<T> << <Grd_Dim_1D, Blk_Dim_1D >> > (exit_wave_ptr, intensity_ptr, pixel_num);

			OL_dev_ptr->set_dsf_sigma(dsf_sigma);
			OL_dev_ptr->set_ssf_sigma(ssf_beta);
		}
		/******************************************************************/
		void PCTF_TEM(const eTemporal_Spatial_Incoh &temporal_spatial_incoh, T_complex<T>* wave_ptr, T* real_ptr, T* imag_ptr)
		{
			T dsf_sigma = OL_dev_ptr->dsf_sigma;
			T ssf_beta = OL_dev_ptr->ssf_beta;
			if (temporal_spatial_incoh == eTSI_Temporal)
			{
				OL_dev_ptr->set_ssf_sigma(0);
			}
			if (temporal_spatial_incoh == eTSI_Spatial)
			{
				OL_dev_ptr->set_dsf_sigma(0);
			}
			apply_PCTF_kernel<T> << <Grd_Dim_2D, Blk_Dim_2D >> > (*grid_dev_ptr, *OL_dev_ptr, wave_ptr, exit_wave_ptr);
			fft_2d_ptr->inverse(exit_wave_ptr);
			wave_spliter<T> << <Grd_Dim_1D, Blk_Dim_1D >> > (exit_wave_ptr, real_ptr, imag_ptr, pixel_num);

			OL_dev_ptr->set_dsf_sigma(dsf_sigma);
			OL_dev_ptr->set_ssf_sigma(ssf_beta);
		}
		void TCC_TEM(const eTemporal_Spatial_Incoh &temporal_spatial_incoh, T_complex<T>* wave_ptr, T* intensity_ptr)
		{
			T c_10_0 = OL_dev_ptr->c_10;
			cudaMemset(intensity_ptr, 0, sizeof(T)*pixel_num);
			switch (temporal_spatial_incoh)
			{
			case eTSI_Temporal_Spatial:	// Temporal and Spatial
			{
				for (auto i = 0; i < qs_ptr->size(); i++)
				{
					for (auto j = 0; j < qt_ptr->size(); j++)
					{
						auto c_10 = OL_dev_ptr->dsf_iehwgd*qt_ptr->x[j] + c_10_0;
						OL_dev_ptr->set_defocus(c_10);
						apply_CTF_kernel<T> << <Grd_Dim_2D, Blk_Dim_2D >> > (*grid_dev_ptr, *OL_dev_ptr, qs_ptr->x[i], qs_ptr->y[i], wave_ptr, exit_wave_ptr);
						fft_2d_ptr->inverse(exit_wave_ptr);
						add_scale_square<T> << <Grd_Dim_1D, Blk_Dim_1D >> > (qs_ptr->w[i] * qt_ptr->w[j], exit_wave_ptr, intensity_ptr, pixel_num);
					}
				}
			}
			break;
			case eTSI_Temporal:// Temporal
			{
				for (auto j = 0; j < qt_ptr->size(); j++)
				{
					auto c_10 = OL_dev_ptr->dsf_iehwgd*qt_ptr->x[j] + c_10_0;
					OL_dev_ptr->set_defocus(c_10);
					apply_CTF_kernel<T> << <Grd_Dim_2D, Blk_Dim_2D >> > (*grid_dev_ptr, *OL_dev_ptr, 0.0, 0.0, wave_ptr, exit_wave_ptr);
					fft_2d_ptr->inverse(exit_wave_ptr);
					add_scale_square<T> << <Grd_Dim_1D, Blk_Dim_1D >> > (qt_ptr->w[j], exit_wave_ptr, intensity_ptr, pixel_num);
				}
			}
			break;
			case eTSI_Spatial:// Spatial
			{
				for (auto i = 0; i < qs_ptr->size(); i++)
				{
					apply_CTF_kernel<T> << <Grd_Dim_2D, Blk_Dim_2D >> > (*grid_dev_ptr, *OL_dev_ptr, qs_ptr->x[i], qs_ptr->y[i], wave_ptr, exit_wave_ptr);
					fft_2d_ptr->inverse(exit_wave_ptr);
					add_scale_square<T> << <Grd_Dim_1D, Blk_Dim_1D >> > (qs_ptr->w[i], exit_wave_ptr, intensity_ptr, pixel_num);
				}
			}
			break;
			}
			OL_dev_ptr->set_defocus(c_10_0);
		}
		void TCC_TEM(const eTemporal_Spatial_Incoh &temporal_spatial_incoh, T_complex<T>* wave_ptr, T* real_ptr, T* imag_ptr)
		{
			T c_10_0 = OL_dev_ptr->c_10;
			set_intensity_val_kernel<T> << <Grd_Dim_1D, Blk_Dim_1D >> > (real_ptr, T(0.0), pixel_num);
			set_intensity_val_kernel<T> << <Grd_Dim_1D, Blk_Dim_1D >> > (imag_ptr, T(0.0), pixel_num);
			switch (temporal_spatial_incoh)
			{
			case eTSI_Temporal_Spatial:	// Temporal and Spatial
			{
				for (auto i = 0; i < qs_ptr->size(); i++)
				{
					for (auto j = 0; j < qt_ptr->size(); j++)
					{
						auto c_10 = OL_dev_ptr->dsf_iehwgd*qt_ptr->x[j] + c_10_0;
						OL_dev_ptr->set_defocus(c_10);
						apply_CTF_kernel<T> << <Grd_Dim_2D, Blk_Dim_2D >> > (*grid_dev_ptr, *OL_dev_ptr, qs_ptr->x[i], qs_ptr->y[i], wave_ptr, exit_wave_ptr);
						fft_2d_ptr->inverse(exit_wave_ptr);
						add_scale_components<T> << <Grd_Dim_1D, Blk_Dim_1D >> > (qs_ptr->w[i] * qt_ptr->w[j], exit_wave_ptr, real_ptr, imag_ptr, pixel_num);
					}
				}
			}
			break;
			case  eTSI_Temporal:	// Temporal
			{
				for (auto j = 0; j < qt_ptr->size(); j++)
				{
					auto c_10 = OL_dev_ptr->dsf_iehwgd*qt_ptr->x[j] + c_10_0;
					OL_dev_ptr->set_defocus(c_10);
					apply_CTF_kernel<T> << <Grd_Dim_2D, Blk_Dim_2D >> > (*grid_dev_ptr, *OL_dev_ptr, 0.0, 0.0, wave_ptr, exit_wave_ptr);
					fft_2d_ptr->inverse(exit_wave_ptr);
					add_scale_components<T> << <Grd_Dim_1D, Blk_Dim_1D >> > (qt_ptr->w[j], exit_wave_ptr, real_ptr, imag_ptr, pixel_num);
				}
			}
			break;
			case eTSI_Spatial:	// Spatial
			{
				for (auto i = 0; i < qs_ptr->size(); i++)
				{
					apply_CTF_kernel<T> << <Grd_Dim_2D, Blk_Dim_2D >> > (*grid_dev_ptr, *OL_dev_ptr, qs_ptr->x[i], qs_ptr->y[i], wave_ptr, exit_wave_ptr);
					fft_2d_ptr->inverse(exit_wave_ptr);
					add_scale_components<T> << <Grd_Dim_1D, Blk_Dim_1D >> > (qs_ptr->w[i], exit_wave_ptr, real_ptr, imag_ptr, pixel_num);
				}
			}
			break;
			}
			OL_dev_ptr->set_defocus(c_10_0);
		}
		multislice_params_dev<T> *multisliceparam_ptr;
		Grid_2d_Dev<T>* grid_dev_ptr;
		Lens_Dev<T>* OL_dev_ptr;
		FFT_Dev<T> *fft_2d_ptr;
		T_complex<T>* exit_wave_ptr;
	private:
		size_t pixel_num;
		Quad_1d<T> *qt_ptr;
		Quad_2d<T> *qs_ptr;
		dim3 Blk_Dim_2D, Grd_Dim_2D;
		dim3 Blk_Dim_1D, Grd_Dim_1D;
	};

}
#endif
