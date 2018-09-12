#ifndef DEVICE_BASIC_FUNCTIONS_H
#define DEVICE_BASIC_FUNCTIONS_H
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_basic_def.cuh"
#include "grid_dev.cuh"
#include "lens_dev.cuh"
namespace cudaEM
{
	template <class T>
	__device__ __forceinline__
	T_complex<T> complex_Mul(const T_complex<T> &compx1, const T_complex<T> &compx2)
	{
		T_complex<T> tempt;
		tempt.x = compx1.x*compx2.x - compx1.y*compx2.y;
		tempt.y = compx1.x*compx2.y + compx1.y*compx2.x;
		return tempt;
	}
	template <class T>
	__device__ __forceinline__
	T complex_Norm(const T_complex<T> &wave)
	{
		T tempt;
		T real_val = wave.x;
		T imag_val = wave.y;
		tempt = real_val*real_val + imag_val*imag_val;
		return tempt;
	}
	template <class T>
	__device__ __forceinline__
	T_complex<T> polar(const T &amplitude, const T &theta)
	{
		T_complex<T> tempt;
		tempt.x = amplitude*cos(theta);
		tempt.y = amplitude*sin(theta);
		return tempt;
	}
	template <class T>
	__device__ __forceinline__
	T_complex<T> euler(const T &x)
	{
		T_complex<T> tempt;
		tempt.x =cos(x);
		tempt.y =sin(x);
		return tempt;
	}
	template <class T>
	__device__ __forceinline__
	int unrolledBinarySearch_c_nR(const T &x0, const T *x)
	{
		int i0 = 0, ie = c_nR - 1;
		int im = (i0 + ie) >> 1; // divide by 2
		if (x0 < x[im]) ie = im; else i0 = im; // 64
		im = (i0 + ie) >> 1; 	// divide by 2
		if (x0 < x[im]) ie = im; else i0 = im; // 32
		im = (i0 + ie) >> 1; 	// divide by 2
		if (x0 < x[im]) ie = im; else i0 = im; // 16
		im = (i0 + ie) >> 1; 	// divide by 2
		if (x0 < x[im]) ie = im; else i0 = im; // 8
		im = (i0 + ie) >> 1; 	// divide by 2
		if (x0 < x[im]) ie = im; else i0 = im; // 4
		im = (i0 + ie) >> 1; 	// divide by 2
		if (x0 < x[im]) ie = im; else i0 = im; // 2
		im = (i0 + ie) >> 1; 	// divide by 2
		if (x0 < x[im]) ie = im; else i0 = im; // 1
		return i0;
	}
	template <class T, class TAtom>
	__device__ __forceinline__
	T get_value_Cubic_Spline(const T &R2, const TAtom &atom)
	{
		const int ix = unrolledBinarySearch_c_nR<T>(R2, atom.R2_dev);
		const T dx = R2 - atom.R2_dev[ix];
		return (((atom.c3_dev[ix] * dx + atom.c2_dev[ix])*dx + atom.c1_dev[ix])*dx + atom.c0_dev[ix]);
	}
	template <class TAtom>
	__device__ __forceinline__
	void get_cubic_poly_coef(const int &iR, TAtom &atom)
	{
		auto idR = 1.0 / (atom.R2_dev[iR + 1] - atom.R2_dev[iR]);
		auto V = atom.c0_dev[iR];
		auto Vn = atom.c0_dev[iR + 1];
		auto dV = atom.c1_dev[iR];
		auto dVn = atom.c1_dev[iR + 1];
		auto m = (Vn - V)*idR;
		auto n = dV + dVn;
		atom.c2_dev[iR] = (3.0*m - n - dV)*idR;
		atom.c3_dev[iR] = (n - 2.0*m)*idR*idR;
	}
	template <class T>
	__device__ __forceinline__
	void apply_tapering_dev(const T &x_tap, const T &alpha, const T &x, T &y, T &dy)
	{
		if (x_tap < x)
		{
			T tap, dtap;
			sincos(alpha*(x - x_tap), &dtap, &tap);
			dy = dy*tap - alpha*y*dtap;
			y *= tap;
		}
	}
	template <class T>
	__device__ __forceinline__ 
	void Vr_dVrir_Doyle_dev(const T &r, T *cl, T *cnl, T &Vr, T &dVrir)
	{
		T r2 = r*r;

		T Vr0 = cl[0] * exp(-cnl[0] * r2);
		T Vr1 = cl[1] * exp(-cnl[1] * r2);
		T Vr2 = cl[2] * exp(-cnl[2] * r2);
		T Vr3 = cl[3] * exp(-cnl[3] * r2);

		Vr = Vr0 + Vr1 + Vr2 + Vr3;
		dVrir = -2 * (cnl[0] * Vr0 + cnl[1] * Vr1 + cnl[2] * Vr2 + cnl[3] * Vr3);
	}

	template <class T>
	__device__ __forceinline__ 
	void Vr_dVrir_Peng_dev(const T &r, T *cl, T *cnl, T &Vr, T &dVrir)
	{
		T r2 = r*r;

		T Vr0 = cl[0] * exp(-cnl[0] * r2);
		T Vr1 = cl[1] * exp(-cnl[1] * r2);
		T Vr2 = cl[2] * exp(-cnl[2] * r2);
		T Vr3 = cl[3] * exp(-cnl[3] * r2);
		T Vr4 = cl[4] * exp(-cnl[4] * r2);

		Vr = Vr0 + Vr1 + Vr2 + Vr3 + Vr4;
		dVrir = -2 * (cnl[0] * Vr0 + cnl[1] * Vr1 + cnl[2] * Vr2 + cnl[3] * Vr3 + cnl[4] * Vr4);
	}

	template <class T>
	__device__ __forceinline__ 
	void Vr_dVrir_Kirkland_dev(const T &r, T *cl, T *cnl, T &Vr, T &dVrir)
	{
		T ir = 1 / r;
		T r2 = r*r;

		T Vr0 = cl[0] * exp(-cnl[0] * r)*ir;
		T Vr1 = cl[1] * exp(-cnl[1] * r)*ir;
		T Vr2 = cl[2] * exp(-cnl[2] * r)*ir;
		T Vr3 = cl[3] * exp(-cnl[3] * r2);
		T Vr4 = cl[4] * exp(-cnl[4] * r2);
		T Vr5 = cl[5] * exp(-cnl[5] * r2);

		Vr = Vr0 + Vr1 + Vr2 + Vr3 + Vr4 + Vr5;
		dVrir = -(Vr0*(cnl[0] + ir) + Vr1*(cnl[1] + ir) + Vr2*(cnl[2] + ir) + 2 * r*(cnl[3] * Vr3 + cnl[4] * Vr4 + cnl[5] * Vr5)) / r;
	}

	template <class T>
	__device__ __forceinline__ 
	void Vr_dVrir_Weickenmeier_dev(const T &r, T *cl, T *cnl, T &Vr, T &dVrir)
	{
		T r2 = r*r;
		T Vr0 = cl[0] * erfc(cnl[0] * r);
		T Vr1 = cl[1] * erfc(cnl[1] * r);
		T Vr2 = cl[2] * erfc(cnl[2] * r);
		T Vr3 = cl[3] * erfc(cnl[3] * r);
		T Vr4 = cl[4] * erfc(cnl[4] * r);
		T Vr5 = cl[5] * erfc(cnl[5] * r);
		Vr = (Vr0 + Vr1 + Vr2 + Vr3 + Vr4 + Vr5) / r;
		dVrir = 2 * (cl[0] * cnl[0] * exp(-cnl[0] * cnl[0] * r2) + cl[1] * cnl[1] * exp(-cnl[1] * cnl[1] * r2) + cl[2] * cnl[2] * exp(-cnl[2] * cnl[2] * r2) +
			cl[3] * cnl[3] * exp(-cnl[3] * cnl[3] * r2) + cl[4] * cnl[4] * exp(-cnl[4] * cnl[4] * r2) + cl[5] * cnl[5] * exp(-cnl[5] * cnl[5] * r2)) / 1.772453850905516027298;
		dVrir = -(dVrir + Vr) / r2;
	}

	template <class T>
	__device__ __forceinline__ 
	void Vr_dVrir_Lobato_dev(const T &r, T *cl, T *cnl, T &Vr, T &dVrir)
	{
		T cnl0r = cnl[0] * r;
		T cnl1r = cnl[1] * r;
		T cnl2r = cnl[2] * r;
		T cnl3r = cnl[3] * r;
		T cnl4r = cnl[4] * r;

		T Vr0 = cl[0] * exp(-cnl0r);
		T Vr1 = cl[1] * exp(-cnl1r);
		T Vr2 = cl[2] * exp(-cnl2r);
		T Vr3 = cl[3] * exp(-cnl3r);
		T Vr4 = cl[4] * exp(-cnl4r);

		Vr = Vr0*(2 / cnl0r + 1) + Vr1*(2 / cnl1r + 1) + Vr2*(2 / cnl2r + 1) + Vr3*(2 / cnl3r + 1) + Vr4*(2 / cnl4r + 1);
		dVrir = -(Vr + Vr0*(cnl0r + 1) + Vr1*(cnl1r + 1) + Vr2*(cnl2r + 1) + Vr3*(cnl3r + 1) + Vr4*(cnl4r + 1)) / (r*r);
	}
	template <class T>
	__device__ __forceinline__
	void Vr_dVrir_dev(ePotential_Type potential_type, const T &r, T *cl, T *cnl, T f, T &Vr, T &dVrir)
	{
		switch (potential_type)
		{
		case ePT_Doyle_0_4:
			Vr_dVrir_Doyle_dev<T>(r, cl, cnl, Vr, dVrir);
			break;
		case ePT_Peng_0_4:
			Vr_dVrir_Peng_dev<T>(r, cl, cnl, Vr, dVrir);
			break;
		case ePT_Peng_0_12:
			Vr_dVrir_Peng_dev<T>(r, cl, cnl, Vr, dVrir);
			break;
		case ePT_Kirkland_0_12:		
			Vr_dVrir_Kirkland_dev<T>(r, cl, cnl, Vr, dVrir);
			break;
		case ePT_Weickenmeier_0_12:
			Vr_dVrir_Weickenmeier_dev<T>(r, cl, cnl, Vr, dVrir);
			break;
		case ePT_Lobato_0_12:
			Vr_dVrir_Lobato_dev<T>(r, cl, cnl, Vr, dVrir);
			break;
		}
		Vr *= f;
		dVrir *= f;
	}
	template <class T>
	__device__ __forceinline__
	T coherent_phase_shift(const T &gx, const T &gy, const Lens_Dev<T> &lens)
	{
		auto g2 = gx*gx + gy*gy;
		auto g4 = g2*g2;
		auto g6 = g4*g2;
		T phase_shift = lens.eval_c_10(g2) + lens.eval_c_30(g4) + lens.eval_c_50(g6);
		if (lens.is_phi_required())
		{
			auto g = sqrt(g2);
			auto g3 = g2*g;
			auto g5 = g4*g;
			auto phi = atan2(gy, gx);
			phase_shift += lens.eval_m(phi) + lens.eval_c_12(g2, phi);
			phase_shift += lens.eval_c_21_c_23(g3, phi) + lens.eval_c_32_c_34(g4, phi);
			phase_shift += lens.eval_c_41_c_43_c_45(g5, phi) + lens.eval_c_52_c_54_c_56(g6, phi);
		}
		return phase_shift;
	}
	template <class T>
	__device__ __forceinline__
	T_complex<T> incoherent_phase_shift(const T &gx, const T &gy, const Lens_Dev<T> &lens)
	{
		auto g2 = gx*gx + gy*gy;
		T_complex<T> v; v.x = T(0.0); v.y = T(0.0);
		if ((lens.g2_min <= g2) && (g2 < lens.g2_max))
		{
			/***************Compute the Chi***************/
			auto g4 = g2*g2;
			auto g6 = g4*g2;
			auto phase_shift = lens.eval_c_10(g2) + lens.eval_c_30(g4) + lens.eval_c_50(g6);
			if (lens.is_phi_required())
			{
				auto g = sqrt(g2);
				auto g3 = g2*g;
				auto g5 = g4*g;
				auto phi = atan2(gy, gx);
				phase_shift += lens.eval_m(phi) + lens.eval_c_12(g2, phi);
				phase_shift += lens.eval_c_21_c_23(g3, phi) + lens.eval_c_32_c_34(g4, phi);
				phase_shift += lens.eval_c_41_c_43_c_45(g5, phi) + lens.eval_c_52_c_54_c_56(g6, phi);
			}
			T c = C_PI_DEV*lens.ssf_beta*lens.dsf_iehwgd;
			T u = 1.0 + 2 * c*c*g2;
			c = C_PI_DEV*lens.dsf_iehwgd*lens.lambda*g2;
			T temp_inc = 0.5*c*c;
			c = C_PI_DEV*lens.ssf_beta*(lens.c_30*lens.lambda_squared*g2 - lens.c_10);
			T spa_inc = c*c*g2;
			T st_inc = exp(-(spa_inc + temp_inc) / u) / sqrt(u);
			v = polar<T>(st_inc, phase_shift);
			return v;
		}
		return v;
	}
}
#endif