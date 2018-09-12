#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <complex>
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include "device_basic_def.cuh"
namespace cudaEM
{
	template <class T>
	struct FFT_Dev
	{
	public:
		FFT_Dev() : plan_dev(0), plan_many_dev(0)
		{
			//determine if the type is float.
			float_type = std::is_same<T, float>::value;
		}

		~FFT_Dev()
		{
			destroy_plan();
		}

		void cleanup()
		{
			destroy_plan();
		}

		void destroy_plan()
		{
			if (plan_dev == 0)
			{
				return;
			}
			cudaDeviceSynchronize();
			cufftDestroy(plan_dev);
			cufftDestroy(plan_many_dev);
			plan_dev = 0;
			plan_many_dev = 0;
		}

		void create_plan_1d(const int &nx, int nThread = 1)
		{
			destroy_plan();
			if (float_type)
			{
				cufftPlan1d(&plan_dev, nx, CUFFT_C2C, 1);
			}
			else
			{
				cufftPlan1d(&plan_dev, nx, CUFFT_Z2Z, 1);
			}
		}
		void create_plan_2d(const int &ny, const int &nx)
		{
			destroy_plan();
			row = ny;
			col = nx;
			if (float_type)
			{
				cufftPlan2d(&plan_dev, nx, ny, CUFFT_C2C);
			}
			else
			{
				cufftPlan2d(&plan_dev, nx, ny, CUFFT_Z2Z);
			}
		}
		void create_plan_2d_many(const int &ny, const int &nx, const int &nframes)
		{
			destroy_plan();

			int rank = 2;						// Dimensionality of the transform (1, 2, or 3). 
			int n[] = { ny, nx };				// 1d transforms of length nx*ny
			int idist = ny*nx, odist = ny*nx;	// distance between two successive input elements in the least significant
			int istride = 1, ostride = 1;		// distance between two elements in the same column
			int *inembed = n, *onembed = n;		// Pointer of size rank that indicates the storage dimensions
			row = ny;
			col = nx;
			if (float_type)
			{
				cufftPlan2d(&plan_dev, nx, ny, CUFFT_C2C);
				cufftPlanMany(&plan_many_dev, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, nframes);
			}
			else
			{
				cufftPlanMany(&plan_many_dev, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_Z2Z, nframes);
				cufftPlan2d(&plan_dev, nx, ny, CUFFT_Z2Z);
			}

		}
		void forward(T_complex<T> *M_io)
		{
			forward(M_io, M_io);
		}
		void inverse(T_complex<T> *M_io)
		{
			inverse(M_io, M_io);
		}
		void forward_batch(T_complex<T> *M_io)
		{
			forward_batch(M_io, M_io);
		}
		void inverse_batch(T_complex<T> *M_io)
		{
			inverse_batch(M_io, M_io);
		}
		void forward(T_complex<T> *M_i, T_complex<T> *M_o)
		{
			cufftExecC2C(plan_dev, M_i, M_o, CUFFT_FORWARD);
		}
		void inverse(T_complex<T> *M_i, T_complex<T> *M_o)
		{
			cufftExecC2C(plan_dev, M_i, M_o, CUFFT_INVERSE);
		}
		void forward_batch(T_complex<T> *M_i, T_complex<T> *M_o)
		{
			cufftExecC2C(plan_many_dev, M_i, M_o, CUFFT_FORWARD);
		}
		void inverse_batch(T_complex<T> *M_i, T_complex<T> *M_o)
		{
			cufftExecC2C(plan_many_dev, M_i, M_o, CUFFT_INVERSE);
		}
	private:
		cufftHandle plan_dev;
		cufftHandle plan_many_dev;
		int row;
		int col;
		bool float_type;
	};
}