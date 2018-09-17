#ifndef GRID_DEV_H
#define GRID_DEV_H
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include "device_basic_def.cuh"
namespace cudaEM
{

	template <class T>
	struct Grid_2d_Dev
	{
		/******************Basic integer settings**********************/
		int col, row;           // The pixel number along x and y direction.
		int col_h, row_h;      // half number
		T x_l, y_l, dz;         // physical length
		bool bwl; 		        // Band-width limit
		T x_dr, y_dr; 	        // x, y-sampling in real space
		T x_dg, y_dg; 	        // x, y-sampling in reciprocal space
		T gl2_max; 		        // Squared of the maximum limited frequency
		T alpha;		        // 1/(1+exp(alpha*(x^2-x_c^2)))
		Grid_2d_Dev() : col(0), row(0), col_h(0), row_h(0),
			x_l(0), y_l(0), dz(0), bwl(true),
			x_dr(0), y_dr(0), x_dg(0), y_dg(0), gl2_max(0) {}

		Grid_2d_Dev(int col_i, int row_i)
		{
			x_l = col_i;
			y_l = row_i;
			set_input_data(col_i, row_i, x_l, y_l);
		}

		Grid_2d_Dev(int col_i, int row_i, T x_l_i, T y_l_i, T dz_i = 0.5, bool bwl_i = false)
		{
			set_input_data(col_i, row_i, x_l_i, y_l_i, dz_i, bwl_i);
		}
		/*********************host code for Grid class*****************************/
		inline void set_input_data(int col_i, int row_i, T x_l_i, T y_l_i, T dz_i = 0.5,
			bool bwl_i = false)
		{
			col = col_i;
			row = row_i;
			col_h = col / 2;
			row_h = row / 2;
			x_l = x_l_i;
			y_l = y_l_i;
			dz = dz_i;
			bwl = bwl_i;
			x_dr = Div(x_l, col);
			y_dr = Div(y_l, row);
			x_dg = Div(T(1.0), x_l);
			y_dg = Div(T(1.0), y_l);
			gl2_max = std::pow(gl_max_bwl(), 2);
			T dg0 = 0.25, fg0 = 1e-02;
			alpha = std::log(1.0 / fg0 - 1.0) / (std::pow(gl_max_bwl() + dg0, 2) - gl2_max);
		}
		template <class HGrid>
		void assign(HGrid &grid_2d)
		{
			set_input_data(grid_2d.col, grid_2d.row, grid_2d.x_l, grid_2d.y_l, grid_2d.dz, grid_2d.bwl);
		}
		template <class HGrid>
		void operator=(HGrid &grid_2d)
		{
			assign(grid_2d);
		}
		/**************minimum and maximum values*****************/
		inline int Col_Row_min() const
		{
			return std::min(col, row);
		}
		inline int Col_Row_max() const
		{
			return std::max(col, row);
		}
		inline T x_y_min() const
		{
			return std::min(x_l, y_l);
		}
		inline T x_y_max() const
		{
			return std::max(x_l, y_l);
		}
		inline T g_max() const
		{
			return std::fmin(static_cast<T>(col_h)*x_dg, static_cast<T>(row_h)*y_dg);
		}
		inline T g_max_squared() const
		{
			return std::pow(g_max(), 2);
		}
		inline T gl_max_bwl() const
		{
			return 2.0*g_max() / 3.0;
		}
		inline T dr_min() const
		{
			return std::fmin(x_dr, y_dr);
		}
		inline T dg_min() const
		{
			return std::fmin(x_dg, y_dg);
		}
		inline T exp_factor_x_pos(const T &x_pos) const
		{
			return -c_2Pi*(x_pos - 0.5*x_l);
		}
		inline T exp_factor_y_pos(const T &y_pos) const
		{
			return -c_2Pi*(y_pos - 0.5*y_l);
		}
		/***********************************************************************************/
		inline std::size_t Grid_size() const
		{
			return static_cast<std::size_t>(col)*static_cast<std::size_t>(row);
		}
		/******************device function running on GPU kernels****************************/
		__device__ T Grid_size_float()
		{
			return T(col)*T(row);
		}
		__device__ T Grid_size_inverse()
		{
			return T(1.0)/T(col*row);
		}
		__device__ size_t row_major_idx(const int &col_idx, const int &row_idx) const
		{
			return (size_t(row_idx*col) + size_t(col_idx));
		}
		__device__ size_t col_major_idx(const int &col_idx, const int &row_idx) const
		{
			return (col_idx*size_t(row) + row_idx);
		}
		/********************************************************************/
		__device__ T X_distance(const int &col_idx, T x_pos = T()) const
		{
			return (col_idx*x_dr - x_pos);
		}
		__device__ T Y_distance(const int &row_idx, T y_pos = T()) const
		{
			return (row_idx*y_dr - y_pos);
		}
		__device__ T X_distance_square(const int &col_idx, T x_pos = T()) const
		{
			return pow(X_distance(col_idx, x_pos), 2);
		}
		__device__ T Y_distance_square(const int &row_idx, T y_pos = T()) const
		{
			return pow(Y_distance(row_idx, y_pos), 2);
		}
		__device__ T Distance_square(const int &col_idx, const int &row_idx, T x_pos = T(), T y_pos = T()) const
		{
			return (X_distance_square(col_idx, x_pos) + Y_distance_square(row_idx, y_pos));
		}
		__device__ T Distance(const int &col_idx, const int &row_idx, T x_pos = T(), T y_pos = T()) const
		{
			return sqrt(Distance_square(col_idx, row_idx, x_pos, y_pos));
		}
		/********************Reciprocal space shift*********************************/
		__device__ int Col_fft_shift(const int &col_idx) const
		{
			return (col_idx < col_h) ? col_idx : (col_idx - col);
		}
		__device__ int Row_fft_shift(const int &row_idx) const
		{
			return (row_idx < row_h) ? row_idx : (row_idx - row);
		}
		__device__ T gx_fft_shift(const int &col_idx, T gx_pos = T()) const
		{
			return (Col_fft_shift(col_idx)*x_dg - gx_pos);
		}
		__device__ T gy_fft_shift(const int &row_idx, T gy_pos = T()) const
		{
			return (Row_fft_shift(row_idx)*y_dg - gy_pos);
		}
		__device__ T gx_fft_shift_squared(const int &col_idx, T gx_pos = T()) const
		{
			return pow(gx_fft_shift(col_idx, gx_pos), 2);
		}
		__device__ T gy_fft_shift_squared(const int &row_idx, T gy_pos = T()) const
		{
			return pow(gy_fft_shift(row_idx, gy_pos), 2);
		}
		__device__ T g_fft_shift_squared(const int &col_idx, const int &row_idx, T gx_pos = T(), T gy_pos = T()) const
		{
			return (gx_fft_shift_squared(col_idx, gx_pos) + gy_fft_shift_squared(row_idx, gy_pos));
		}
		__device__ T g_fft_shift(const int &col_idx, const int &row_idx, T gx_pos = T(), T gy_pos = T()) const
		{
			return sqrt(g_fft_shift_squared(col_idx, row_idx, gx_pos, gy_pos));
		}
		/***********************Band width limitation settings**************************/
		__device__ T bwl_factor_shift(const int &col_idx, const int &row_idx) const
		{
			return 1.0 / (1.0 + exp(alpha*(g_fft_shift_squared(col_idx, row_idx) - gl2_max)));
		}
		/************************Generate the dimensional of CUDA kernels***********************/
		dim3 cuda_grid_2D(const dim3 Grid_max = dim3(0, 0))
		{
			dim3 grid_dim;
			grid_dim = dim3((row + c_thrnxny - 1) / c_thrnxny, (col + c_thrnxny - 1) / c_thrnxny);
			if (Grid_max.x != 0)
			{
				grid_dim.x = std::min(Grid_max.x, grid_dim.x);
			}
			if (Grid_max.y != 0)
			{
				grid_dim.y = std::min(Grid_max.y, grid_dim.y);
			}
			return grid_dim;
		}
		dim3 cuda_grid_2D_h(const dim3 Grid_max = dim3(0, 0))
		{
			dim3 grid_dim;
			grid_dim = dim3((row_h + c_thrnxny - 1) / c_thrnxny, (col_h + c_thrnxny - 1) / c_thrnxny);
			if (Grid_max.x != 0)
			{
				grid_dim.x = std::min(Grid_max.x, grid_dim.x);
			}
			if (Grid_max.y != 0)
			{
				grid_dim.y = std::min(Grid_max.y, grid_dim.y);
			}
			return grid_dim;
		}
	};
	template<class T>
	struct Grid_square_Dev : Grid_2d_Dev<T>
	{
		inline Grid_square_Dev(): Grid_2d_Dev<T>() {}
		Grid_square_Dev(int col_i, T x_l_i)
		{
			this->set_input_data(col_i, col_i, x_l_i, x_l_i);
		}
		Grid_square_Dev(int col_i, T x_l_i, T dz_i = 0.5, bool bwl_i = false)
		{
			this->set_input_data(col_i, col_i, x_l_i, x_l_i, dz_i, bwl_i);
		}
		inline void set_square_input_data(int col_i, T x_l_i, T dz_i = 0.5, bool bwl_i = false)
		{
			this->set_input_data(col_i, col_i, x_l_i, x_l_i, dz_i, bwl_i);
		}
		template <class HGrid>
		inline void assign(HGrid &grid_2d)
		{
			int col_i = grid_2d.Col_Row_max();
			T   x_l_i = grid_2d.x_y_max();
			this->set_input_data(col_i, col_i, x_l_i, x_l_i, grid_2d.dz, grid_2d.bwl);
		}
		template <class HGrid>
		void operator=(HGrid &grid_2d)
		{
			assign(grid_2d);
		}
	};
}
#endif