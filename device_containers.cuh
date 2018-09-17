#ifndef DEVICE_BASIS_H
#define DEVICE_BASIS_H
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_basic_def.cuh"
namespace cudaEM
{
	/***************The coefficient storing the scattering coefficients in device******************/
	template <class T>
	struct LNL_Coef_Dev
	{
		LNL_Coef_Dev() : m_size(0) 
		{
			cl_dev = 0;
			cnl_dev = 0;
		}
		~LNL_Coef_Dev()
		{
			free_data();
		}
		template <class HCoef>
		void operator = (HCoef &rhs)
		{
			assign(rhs);
		}
		template <class HCoef>
		void assign(HCoef &rhs)
		{
			free_data();
			m_size = rhs.size();
			std::size_t memsize = sizeof(T)*m_size;
			if (cudaSuccess!= cudaMalloc((void **)&cl_dev, 2 * memsize))
			{
				std::cout << "unable to allocate memory for LNL coeff" << std::endl;
				return;
			}
			cnl_dev = cl_dev + m_size;
			cudaMemcpy(cl_dev, rhs.cl.data(), memsize, cudaMemcpyHostToDevice);
			cudaMemcpy(cnl_dev, rhs.cnl.data(), memsize, cudaMemcpyHostToDevice);
		}
		inline void free_data()
		{
			if (cl_dev!=0)
			{
				cudaFree(cl_dev);
				cl_dev = 0;
				cnl_dev = 0;
			}
		}
		std::size_t m_size;
		T *cl_dev;
		T *cnl_dev;
	};
	template <class T>
	struct Quad_1d_Dev
	{
		Quad_1d_Dev() : m_size(0) 
		{
			x_dev = 0;
			w_dev = 0;
		}
		~Quad_1d_Dev()
		{
			free_data();
		}
		template <class HQuad1>
		void operator = (HQuad1 &q1)
		{
			assign(q1);
		}
		template <class HQuad1>
		void assign(HQuad1 &q1)
		{
			free_data();
			m_size = q1.size();
			std::size_t memsize = sizeof(T)*m_size;
			if (cudaSuccess != cudaMalloc((void **)&x_dev, 2 * memsize))
			{
				std::cout << "unable to allocate memory for quadrature" << std::endl;
				return;
			}
			w_dev = x_dev + m_size;
			cudaMemcpy(x_dev, q1.x.data(), memsize, cudaMemcpyHostToDevice);
			cudaMemcpy(w_dev, q1.w.data(), memsize, cudaMemcpyHostToDevice);
		}
		inline void free_data()
		{
			if (x_dev!=0)
			{
				cudaFree(x_dev);
				x_dev = 0;
				w_dev = 0;
			}
		}
		std::size_t m_size;
		T *x_dev;
		T *w_dev;
	};

	template <class T>
	struct Quad_2d_Dev
	{
		Quad_2d_Dev() : m_size(0) 
		{
			x_dev = 0;
			y_dev = 0;
			w_dev = 0;
		}
		~Quad_2d_Dev()
		{
			free_data();
		}
		template <class HQund2>
		void operator = (HQund2 &q2)
		{
			assign(q2);
		}
		template <class HQund2>
		void assign(HQund2 &q2)
		{
			free_data();
			m_size = q2.size();
			std::size_t memsize = sizeof(T)*m_size;
			if (cudaSuccess != cudaMalloc((void **)&x_dev, 3 * memsize))
			{
				std::cout << "unable to allocate memory for quadrature" << std::endl;
				return;
			}
			w_dev = x_dev + m_size;
			y_dev = x_dev + 2*m_size;
			cudaMemcpy(x_dev, q2.x.data(), memsize, cudaMemcpyHostToDevice);
			cudaMemcpy(y_dev, q2.y.data(), memsize, cudaMemcpyHostToDevice);
			cudaMemcpy(w_dev, q2.w.data(), memsize, cudaMemcpyHostToDevice);
		}
		void free_data()
		{
			if (x_dev!=0)
			{
				cudaFree(x_dev);
				x_dev = 0;
				y_dev = 0;
				w_dev = 0;
			}
		}
		std::size_t m_size;
		T *x_dev;
		T *y_dev;
		T *w_dev;
	};
	/************Cubic interpolation coefficients*************/
	template <class T>
	struct CSI_Coef_Dev
	{
		CSI_Coef_Dev() : m_size(0) 
		{
			c0_dev = 0;
			c1_dev = 0;
			c2_dev = 0;
			c3_dev = 0;
		}
		~CSI_Coef_Dev()
		{
			free_data();
		}
		template <class HCI_Coef>
		void operator = (HCI_Coef &ci_coef)
		{
			assign(ci_coef);
		}
		template <class HCI_Coef>
		void assign(HCI_Coef &ci_coef)
		{
			free_data();
			m_size = ci_coef.size();
			std::size_t memsize = sizeof(T)*m_size;
			if (cudaSuccess!= cudaMalloc((void **)&c0_dev, 4 * memsize))
			{
				std::cout << "unable to allocate memory for CSI Data" << std::endl;
				return;
			}
			c1_dev = c0_dev + m_size;
			c2_dev = c0_dev + 2 * m_size;
			c3_dev = c0_dev + 3 * m_size;
			cudaMemcpy(c0_dev, ci_coef.c0.data(), memsize, cudaMemcpyHostToDevice);
			cudaMemcpy(c1_dev, ci_coef.c1.data(), memsize, cudaMemcpyHostToDevice);
			cudaMemcpy(c2_dev, ci_coef.c2.data(), memsize, cudaMemcpyHostToDevice);
			cudaMemcpy(c3_dev, ci_coef.c3.data(), memsize, cudaMemcpyHostToDevice);
		}
		inline void free_data()
		{
			if (c0_dev!=0)
			{
				cudaFree(c0_dev);
				c0_dev = 0;
				c1_dev = 0;
				c2_dev = 0;
				c3_dev = 0;
			}
			
		}
		std::size_t m_size;
		T* c0_dev; 	// zero coefficient
		T* c1_dev; 	// first coefficient
		T* c2_dev; 	// second coefficient
		T* c3_dev; 	// third coefficient
	};
	/************Basci scattering coefficient for single element************/
	template <class T>
	struct Element_Coeffs_Dev
	{
		Element_Coeffs_Dev() : Z(0), m(0), A(0), rn_e(0), rn_c(0), ra_e(0), ra_c(0), R_min(0), R_max(0), R_tap(0), tap_cf(0) {}
		~Element_Coeffs_Dev()
		{
			cudaFree(R_Cubic_dev_ptr);
		}
		template <class HElement_Coef>
		void assign(HElement_Coef &element_coef)
		{
			//basic information
			Z = element_coef.Z;
			m = element_coef.m;
			A = element_coef.A;
			rn_e = element_coef.rn_e;
			rn_c = element_coef.rn_c;
			ra_e = element_coef.ra_e;
			ra_c = element_coef.ra_c;

			R_min = element_coef.R_min;
			R_max = element_coef.R_max;

			//tapering coefficients
			R_tap = element_coef.R_tap;
			tap_cf = element_coef.tap_cf;

			//Copy the coefficients from host to device.
			feg_coeff_dev = element_coef.feg_coeff;
			fxg_coeff_dev = element_coef.fxg_coeff;
			Pr_coeff_dev = element_coef.Pr_coeff;
			Vr_coeff_dev = element_coef.Vr_coeff;
			VR_coeff_dev = element_coef.VR_coeff;

			//spline coefficients
			VR_Cubic_coeff_dev = element_coef.VR_Cubic_coeff;
			cudaMalloc((void**)&R_Cubic_dev_ptr, 2*sizeof(T)*element_coef.R_Cubic.size());
			R2_Cubic_dev_ptr = R_Cubic_dev_ptr+ element_coef.R_Cubic.size();
			cudaMemcpy(R_Cubic_dev_ptr, element_coef.R_Cubic.data(),
				sizeof(T)*element_coef.R_Cubic.size(), cudaMemcpyHostToDevice);
			cudaMemcpy(R2_Cubic_dev_ptr, element_coef.R2_Cubic.data(),
				sizeof(T)*element_coef.R2_Cubic.size(), cudaMemcpyHostToDevice);
		}

		template <class HElement_Coef>
		void operator=(HElement_Coef &element_coef)
		{
			assign(element_coef);
		}
		// Minimum interaction radius squared
		T R2_min() const { return std::pow(R_min, 2); }

		// Maximum interaction radius squared
		T R2_max() const { return std::pow(R_max, 2); }

		// Tapering radius squared
		T R2_tap() const { return std::pow(R_tap, 2); }
		/*************basic information************/
		int Z; 						// Atomic number
		T m; 						// Atomic mass
		int A; 						// Mass number
		T rn_e; 					// Experimental Nuclear radius
		T rn_c; 					// Calculated Nuclear radius
		T ra_e; 					// Experimental atomic radius
		T ra_c; 					// Calculated atomic radius
		T R_min; 					// Minimum interaction radius
		T R_max; 					// Maximum interaction radius
		T R_tap; 					// Tapering radius
		T tap_cf; 					// Tapering cosine factor
		LNL_Coef_Dev<T> feg_coeff_dev; 		// Electron scattering factor coefficients
		LNL_Coef_Dev<T> fxg_coeff_dev; 		// X-ray scattering factor coefficients
		LNL_Coef_Dev<T> Pr_coeff_dev; 		// Projected_Potential coefficients
		LNL_Coef_Dev<T> Vr_coeff_dev; 		// Projected_Potential coefficients
		LNL_Coef_Dev<T> VR_coeff_dev; 		// Projected potential coefficients
		CSI_Coef_Dev<T> VR_Cubic_coeff_dev;		// Look up table - Projected potential coefficients
		T* R_Cubic_dev_ptr; 			// R
		T* R2_Cubic_dev_ptr; 			// R2
	};
	template <class T>
	struct Combined_CSI_Data_Dev
	{
		Combined_CSI_Data_Dev()
		{
			data_size = 0;
			c0_dev = 0;
			c1_dev = 0;
			c2_dev = 0;
			c3_dev = 0;
		}
		void resize(const std::size_t &data_size)
		{
			this->data_size = data_size;
			cudaMalloc((void **)&c0_dev, 4*data_size * sizeof(T));
			c1_dev = c0_dev + data_size;
			c2_dev = c0_dev + 2*data_size;
			c3_dev = c0_dev + 3*data_size;
		}
		~Combined_CSI_Data_Dev()
		{
			free_data();
		}
		inline void free_data()
		{
			if (c0_dev!=0)
			{
				cudaFree(c0_dev);
				data_size = 0;
				c0_dev = 0;
				c1_dev = 0;
				c2_dev = 0;
				c3_dev = 0;
			}
		}
		std::size_t data_size;
		T* c0_dev; 	    // zero coefficient
		T* c1_dev; 		// first coefficient
		T* c2_dev; 		// second coefficient
		T* c3_dev; 		// third coefficient
	};
	/**********************Pointer set used to compute potential************/
	template <class T>
	struct Combined_Element_Coeff_Ptrs
	{
		//atom position
		T x, y;
		//integration range and the occupation
		T z_middle, occ;
		//Tapering distance
		T R_tap_squared, tap_cf;
		//squared maximum radius and minimum radius
		T R_min_squared, R_max_squared;
		/*****Pointers******/
		T *R2_dev;
		T *c3_dev;
		T *c2_dev;
		T *c1_dev;
		T *c0_dev;
		//linear and nonlinear coefficient pointers
		T *cl_dev;
		T *cnl_dev;
	};
}
#endif