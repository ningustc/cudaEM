#pragma once
#ifndef VECTOR_BASICS_H
#define VECTOR_BASICS_H
#include <random>
#include <cmath>
#include <cstdlib>
namespace cudaEM
{
	template <class T>
	struct r3d;
	template <class T>
	struct Rand_r3d
	{
	public:
		void set_seed(int seed_new)
		{
			gen_u.seed(seed_new);
			rand_u.reset();
			rand_temp.reset();
		}

		r3d<T> operator()()
		{
			return r3d<T>(rand_u(gen_u), rand_u(gen_u), rand_u(gen_u));
		}

		T temp()
		{
			rand_temp(gen_u);
		}

	private:
		std::mt19937_64 gen_u;
		std::uniform_real_distribution<T> rand_u;
		std::uniform_real_distribution<T> rand_temp;
	};
	/******************************2d vector*********************************/
	template <class T>
	struct r2d
	{
		T x;
		T y;

		inline r2d(const T &x_i = T(), const T &y_i = T())
		{
			x = x_i;
			y = y_i;
		}

		template <typename X>
		inline r2d(const r2d<X> &r)
		{
			x = r.x;
			y = r.y;
		}

		template <typename X>
		friend inline r2d<X> operator-(const r2d<X> r);

		inline r2d<T>& operator+=(const r2d<T> r)
		{
			x += r.x;
			y += r.y;
			return *this;
		}
		inline r2d<T>& operator-=(const r2d<T> r)
		{
			x -= r.x;
			y -= r.y;
			return *this;
		}
		inline r2d<T>& operator*=(const T r)
		{
			x *= r;
			y *= r;
			return *this;
		}
		inline r2d<T>& operator/=(const T r)
		{
			x /= r;
			y /= r;
			return *this;
		}
		inline T norm() const
		{
			return x*x + y*y;
		}
		inline T module() const
		{
			return std::sqrt(norm());
		}
		inline void normalized()
		{
			*this /= module();
		}

		template <class TVector>
		inline r2d<T> apply_matrix(const TVector &Rm)
		{
			r2d<T> r_o;
			r_o.x = Rm[0] * x + Rm[2] * y;
			r_o.y = Rm[1] * x + Rm[3] * y;
			return r_o;
		}

		template <class TVector>
		inline r2d<T> rotate(const TVector &Rm, const r2d<T> &p0)
		{
			r2d<T> r(x, y);
			r -= p0;
			return (apply_matrix(Rm) + p0);
		}

	};

	template <class X>
	inline r2d<X> operator-(const r2d<X> r)
	{
		return r2d<X>(-r.x, -r.y);
	}

	template <class X>
	inline r2d<X> operator+(const r2d<X> &lhs, const r2d<X> &rhs)
	{
		return r2d<X>(lhs.x + rhs.x, lhs.y + rhs.y);
	}

	template <class X>
	inline r2d<X> operator+(const r2d<X> &lhs, const X &rhs)
	{
		return r2d<X>(lhs.x + rhs, lhs.y + rhs);
	}

	template <class X>
	inline r2d<X> operator+(const X &lhs, const r2d<X> &rhs)
	{
		return r2d<X>(lhs + rhs.x, lhs + rhs.y);
	}

	template <class X>
	inline r2d<X> operator-(const r2d<X> &lhs, const r2d<X> &rhs)
	{
		return r2d<X>(lhs.x - rhs.x, lhs.y - rhs.y);
	}

	template <class X>
	inline r2d<X> operator-(const r2d<X> &lhs, const X &rhs)
	{
		return r2d<X>(lhs.x - rhs, lhs.y - rhs);
	}

	template <class X>
	inline r2d<X> operator-(const X &lhs, const r2d<X> &rhs)
	{
		return r2d<X>(lhs - rhs.x, lhs - rhs.y);
	}

	template <class X>
	inline r2d<X> operator*(const r2d<X> &lhs, const r2d<X> &rhs)
	{
		return r2d<X>(lhs.x*rhs.x, lhs.y*rhs.y);
	}

	template <class X>
	inline r2d<X> operator*(const r2d<X> &lhs, const X &rhs)
	{
		return r2d<X>(lhs.x*rhs, lhs.y*rhs);
	}

	template <class X>
	inline r2d<X> operator*(const X &lhs, const r2d<X> &rhs)
	{
		return r2d<X>(lhs*rhs.x, lhs*rhs.y);
	}

	template <class X>
	inline r2d<X> operator/(const r2d<X> &lhs, const X &rhs)
	{
		return r2d<X>(lhs.x / rhs, lhs.y / rhs);
	}

	template <class X>
	inline r2d<X> fmin(const r2d<X> lhs, const r2d<X> rhs)
	{
		return r2d<X>(std::fmin(lhs.x, rhs.x), std::fmin(lhs.y, rhs.y));
	}

	template <class X>
	inline r2d<X> fmax(const r2d<X> lhs, const r2d<X> rhs)
	{
		return r2d<X>(std::fmax(lhs.x, rhs.x), std::fmax(lhs.y, rhs.y));
	}

	template <typename X>
	inline X norm(const r2d<X>& r)
	{
		return r.x*r.x + r.y*r.y;
	}

	template <typename X>
	inline X module(const r2d<X>& r)
	{
		return std::sqrt(norm(r));
	}

	template <typename X>
	inline r2d<X> normalized(const r2d<X>& r)
	{
		return r / module(r);
	}
	template <class X>
	inline X dot(const r2d<X> &lhs, const r2d<X> &rhs)
	{
		return lhs.x*rhs.x + lhs.y*rhs.y;
	}

	template <class X>
	inline X angle(const r2d<X> &lhs, const r2d<X> &rhs)
	{
		return std::acos(dot(lhs, rhs) / (lhs.module()*rhs.module()));
	}

	/******************************3d vector*********************************/
	template <class T>
	struct r3d
	{
		T x;
		T y;
		T z;

		r3d(const T &x_i = T(), const T &y_i = T(), const T &z_i = T())
		{
			x = x_i;
			y = y_i;
			z = z_i;
		}

		template <typename X>
		r3d(const r3d<X> &r)
		{
			x = r.x;
			y = r.y;
			z = r.z;
		}

		inline r3d<T>& operator+=(const r3d<T> r)
		{
			x += r.x;
			y += r.y;
			z += r.z;
			return *this;
		}
		inline r3d<T>& operator-=(const r3d<T> r)
		{
			x -= r.x;
			y -= r.y;
			z -= r.z;
			return *this;
		}

		inline r3d<T>& operator*=(const T r)
		{
			x *= r;
			y *= r;
			z *= r;
			return *this;
		}

		inline r3d<T>& operator/=(const T r)
		{
			x /= r;
			y /= r;
			z /= r;
			return *this;
		}

		inline T norm()
		{
			return std::pow(x, 2) + std::pow(y, 2) + std::pow(z, 2);
		}

		inline T module()
		{
			return std::sqrt(norm());
		}

		inline void normalized()
		{
			*this /= module();
		}

		template <class TVector>
		inline r3d<T> apply_matrix(const TVector &Rm)
		{
			r3d<T> r_o;
			r_o.x = Rm[0] * x + Rm[3] * y + Rm[6] * z;
			r_o.y = Rm[1] * x + Rm[4] * y + Rm[7] * z;
			r_o.z = Rm[2] * x + Rm[5] * y + Rm[8] * z;
			return r_o;
		}

		template <class TVector>
		inline r3d<T> rotate(const TVector &Rm, const r3d<T> &p0) const
		{
			r3d<T> r(x, y, z);
			r -= p0;
			return (r.apply_matrix(Rm) + p0);
		}

	};

	template <class X>
	inline r3d<X> operator+(const r3d<X> &lhs, const r3d<X> &rhs)
	{
		return r3d<X>(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
	}

	template <class X>
	inline r3d<X> operator+(const r3d<X> &lhs, const X &rhs)
	{
		return r3d<X>(lhs.x + rhs, lhs.y + rhs, lhs.z + rhs);
	}

	template <class X>
	inline r3d<X> operator+(const X &lhs, const r3d<X> &rhs)
	{
		return r3d<X>(lhs + rhs.x, lhs + rhs.y, lhs + rhs.z);
	}

	template <class X>
	inline r3d<X> operator-(const r3d<X> &lhs, const r3d<X> &rhs)
	{
		return r3d<X>(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
	}

	template <class X>
	inline r3d<X> operator-(const r3d<X> &lhs, const X &rhs)
	{
		return r3d<X>(lhs.x - rhs, lhs.y - rhs, lhs.z - rhs);
	}

	template <class X>
	inline r3d<X> operator-(const X &lhs, const r3d<X> &rhs)
	{
		return r3d<X>(lhs - rhs.x, lhs - rhs.y, lhs - rhs.z);
	}

	template <class X>
	inline r3d<X> operator*(const r3d<X> &lhs, const r3d<X> &rhs)
	{
		return r3d<X>(lhs.x*rhs.x, lhs.y*rhs.y, lhs.z*rhs.z);
	}

	template <class X>
	inline r3d<X> operator*(const r3d<X> &lhs, const X &rhs)
	{
		return r3d<X>(lhs.x*rhs, lhs.y*rhs, lhs.z*rhs);
	}

	template <class X>
	inline r3d<X> operator*(const X &lhs, const r3d<X> &rhs)
	{
		return r3d<X>(lhs*rhs.x, lhs*rhs.y, lhs*rhs.z);
	}

	template <class X>
	inline r3d<X> operator/(const r3d<X> &lhs, const X &rhs)
	{
		return r3d<X>(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs);
	}

	template <class X>
	inline r3d<X> fmin(const r3d<X> lhs, const r3d<X> rhs)
	{
		return r3d<X>(std::fmin(lhs.x, rhs.x), std::fmin(lhs.y, rhs.y), std::fmin(lhs.z, rhs.z));
	}

	template <class X>
	inline r3d<X> fmax(const r3d<X> lhs, const r3d<X> rhs)
	{
		return r3d<X>(std::fmax(lhs.x, rhs.x), std::fmax(lhs.y, rhs.y), std::fmax(lhs.z, rhs.z));
	}

	template <typename X>
	inline X norm(const r3d<X>& r)
	{
		return r.x*r.x + r.y*r.y + r.z*r.z;
	}

	template <typename X>
	inline X module(const r3d<X>& r)
	{
		return std::sqrt(norm(r));
	}

	template <typename X>
	inline r3d<X> normalized(const r3d<X>& r)
	{
		return r / r.module();
	}
	template <class X>
	inline X dot(const r3d<X> &lhs, const r3d<X> &rhs)
	{
		return lhs.x*rhs.x + lhs.y*rhs.y + lhs.z*rhs.z;
	}

	template <class X>
	inline X angle(const r3d<X> &lhs, const r3d<X> &rhs)
	{
		return std::acos(dot(lhs, rhs) / (lhs.module()*rhs.module()));
	}
}
#endif