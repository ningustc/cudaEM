#ifndef INCIDENT_WAVE_H
#define INCIDENT_WAVE_H
#include "device_basic_def.cuh"
#include "multislice_params_dev.cuh"
namespace cudaEM
{
	template <class T>
	class Incident_Wave {
	public:
		Incident_Wave() : multisliceparam_ptr(nullptr){}
		void set_input_data(multislice_params_dev<T> *input_multislice_i)
		{
			multisliceparam_ptr = input_multislice_i;
			fft_2d_ptr = &input_multislice_i->fft_trans_dev;
			grid_dev_ptr = &input_multislice_i->grid_dev;
			CL_dev_ptr = &input_multislice_i->cond_lens_dev;
			Blk_Dim_1D = input_multislice_i->Blk_Dim_1D;
			Grd_Dim_1D = input_multislice_i->Grd_Dim_1D;
			Blk_Dim_2D = input_multislice_i->Blk_Dim_2D;
			Grd_Dim_2D = input_multislice_i->Grd_Dim_2D;
			pixel_num = grid_dev_ptr->Grid_size();
			x_pos_exp_ptr = input_multislice_i->x_pos_exp;
			y_pos_exp_ptr = input_multislice_i->y_pos_exp;
			plane_wave_val.x = T(1.0);
			plane_wave_val.y = T(0.0);
		}

		// x, and y is the beam position.
		void generate_incident_beam(T_complex<T>* wave_ptr, T gxu, T gyu, T x_b, T y_b, T z_init = 0)
		{
			if (multisliceparam_ptr->iw_type==eIWT_Convergent_Wave)
			{
				auto f_0 = CL_dev_ptr->c_10;
				auto f_s = f_0 + z_init;
				CL_dev_ptr->set_defocus(f_s);
				auto x = grid_dev_ptr->exp_factor_x_pos(x_b);
				auto y = grid_dev_ptr->exp_factor_y_pos(y_b);
				Generate_probe<T> << <Grd_Dim_2D, Blk_Dim_2D >> >(*grid_dev_ptr, *CL_dev_ptr, x, y, gxu, gyu, wave_ptr);
				fft_2d_ptr->inverse(wave_ptr);
				CL_dev_ptr->set_defocus(f_0);
			}
			else
			{
				//this is the plane wave case, however, tilted wave is not this case.
				set_wave_val_kernel<T> << <Grd_Dim_1D, Blk_Dim_1D >> > (wave_ptr, plane_wave_val, pixel_num);
			}
		}
		void generate_incident_beams(T_complex<T>* wave_ptr, T gxu, T gyu, int idx_start, int batch_num,T z_init = 0)
		{

			if (multisliceparam_ptr->iw_type == eIWT_Convergent_Wave)
			{
				//for the convergent beam, make the full integration.=
				auto f_0 = CL_dev_ptr->c_10;
				auto f_s = f_0 + z_init;
				CL_dev_ptr->set_defocus(f_s);
				T* x_pos_start = x_pos_exp_ptr + idx_start;
				T* y_pos_start = y_pos_exp_ptr + idx_start;
				// determine the phase shift brought by probe position shift in the reciprocal space
				// the beam is taking the middle as origin.
				Generate_multi_probe<T> << <Grd_Dim_2D, Blk_Dim_2D >> >(*grid_dev_ptr, *CL_dev_ptr, x_pos_start,
					y_pos_start, gxu, gyu, batch_num, pixel_num,wave_ptr);
				fft_2d_ptr->inverse_batch(wave_ptr);
				CL_dev_ptr->set_defocus(f_0);
			}
			else
			{
				//this is the plane wave case, however, tilted wave is not this case. 
				set_wave_val_kernel<T> << <Grd_Dim_1D, Blk_Dim_1D >> > (wave_ptr, plane_wave_val, pixel_num*batch_num);
			}
		}
		void apply_beam_tilt(const T &gxu, const T &gyu, T_complex<T>* wave_ptr_i, T_complex<T>* wave_ptr_o)
		{
			if (!isZero(gxu, gyu))
			{
				exp_r_factor_2d<T> << <Grd_Dim_2D, Blk_Dim_2D >> >(*grid_dev_ptr, c_2Pi*gxu, c_2Pi*gyu, wave_ptr_i, wave_ptr_o);
			}
		}
	private:
		size_t pixel_num;
		T_complex<T> plane_wave_val;
		multislice_params_dev<T> *multisliceparam_ptr;
		Grid_2d_Dev<T>* grid_dev_ptr;
		Lens_Dev<T>* CL_dev_ptr;
		FFT_Dev<T> *fft_2d_ptr;
		dim3 Blk_Dim_2D, Grd_Dim_2D;
		dim3 Blk_Dim_1D, Grd_Dim_1D;
		T* x_pos_exp_ptr;
		T* y_pos_exp_ptr;
	};

}

#endif