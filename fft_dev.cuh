#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <complex>
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
			cufftCreate(&plan_dev);
			if (float_type)
			{
				cufftMakePlan1d(plan_dev, nx, CUFFT_C2C, 1, &workSize);
			}
			else
			{
				cufftMakePlan1d(plan_dev, nx, CUFFT_Z2Z, 1, &workSize);
			}
		}
		void create_plan_2d(const int &ny, const int &nx)
		{
			destroy_plan();
			cufftCreate(&plan_dev);
			row = ny;
			col = nx;
			if (float_type)
			{
				cufftMakePlan2d(plan_dev, nx, ny, CUFFT_C2C, &workSize);
			}
			else
			{
				cufftMakePlan2d(plan_dev, nx, ny, CUFFT_Z2Z, &workSize);
			}
		}
		void create_plan_2d_many(const int &ny, const int &nx, const int &nframes)
		{
			destroy_plan();
			cufftCreate(&plan_dev);
			cufftCreate(&plan_many_dev);
			int rank = 2;						// Dimensionality of the transform (1, 2, or 3). 
			int n[] = { ny, nx };				// 1d transforms of length nx*ny
			int idist = ny*nx, odist = ny*nx;	// distance between two successive input elements in the least significant
			int istride = 1, ostride = 1;		// distance between two elements in the same column
			int *inembed = n, *onembed = n;		// Pointer of size rank that indicates the storage dimensions
			row = ny;
			col = nx;
			if (float_type)
			{
				cufftMakePlan2d(plan_dev, nx, ny, CUFFT_C2C, &workSize);
				cufftMakePlanMany(plan_many_dev, rank, n, inembed, istride, idist,
					onembed, ostride, odist, CUFFT_C2C, nframes, &workSize_many);
			}
			else
			{
				cufftMakePlanMany(plan_many_dev, rank, n, inembed, istride, idist,
					onembed, ostride, odist, CUFFT_Z2Z, nframes, &workSize_many);
				cufftMakePlan2d(plan_dev, nx, ny, CUFFT_Z2Z, &workSize);
			}
		}
		void attach_callback_batch(void **h_storeCallbackPtr,
			cufftXtCallbackType_t callback_type, void **callback_infos = 0)
		{
			cufftResult callback_result =
				cufftXtSetCallback(plan_many_dev, h_storeCallbackPtr,
					callback_type, callback_infos);
			if (callback_result != CUFFT_SUCCESS)
			{
				std::cout << "please compile the code using static library"
					<< std::endl;
			}
		}
		void attach_callback(void **h_storeCallbackPtr,
			cufftXtCallbackType_t callback_type, void **callback_infos = 0)
		{
			cufftResult callback_result =
				cufftXtSetCallback(plan_dev, h_storeCallbackPtr,
					callback_type, callback_infos);
			if (callback_result != CUFFT_SUCCESS)
			{
				std::cout << "please compile the code using static library"
					<< std::endl;
			}
		}
		void deattach_callback(cufftXtCallbackType_t callback_type)
		{
			cufftXtClearCallback(plan_dev, callback_type);
		}
		void deattach_callback_batch(cufftXtCallbackType_t callback_type)
		{
			cufftXtClearCallback(plan_many_dev, callback_type);
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
			forward_trans(M_i, M_o);
		}
		void inverse(T_complex<T> *M_i, T_complex<T> *M_o)
		{
			inverse_trans(M_i, M_o);
		}
		void forward_batch(T_complex<T> *M_i, T_complex<T> *M_o)
		{
			forward_trans_many(M_i, M_o);
		}
		void inverse_batch(T_complex<T> *M_i, T_complex<T> *M_o)
		{
			inverse_trans_many(M_i, M_o);
		}
	private:
		inline void forward_trans(cufftComplex*M_i, cufftComplex*M_o)
		{
			cufftExecC2C(plan_dev, M_i, M_o, CUFFT_FORWARD);
		}
		inline void forward_trans(cufftDoubleComplex*M_i, cufftDoubleComplex*M_o)
		{
			cufftExecZ2Z(plan_dev, M_i, M_o, CUFFT_FORWARD);
		}
		inline void inverse_trans(cufftComplex*M_i, cufftComplex*M_o)
		{
			cufftExecC2C(plan_dev, M_i, M_o, CUFFT_INVERSE);
		}
		inline void inverse_trans(cufftDoubleComplex*M_i, cufftDoubleComplex*M_o)
		{
			cufftExecZ2Z(plan_dev, M_i, M_o, CUFFT_INVERSE);
		}
		inline void forward_trans_many(cufftComplex*M_i, cufftComplex*M_o)
		{
			cufftExecC2C(plan_many_dev, M_i, M_o, CUFFT_FORWARD);
		}
		inline void forward_trans_many(cufftDoubleComplex*M_i, cufftDoubleComplex*M_o)
		{
			cufftExecZ2Z(plan_many_dev, M_i, M_o, CUFFT_FORWARD);
		}
		inline void inverse_trans_many(cufftComplex*M_i, cufftComplex*M_o)
		{
			cufftExecC2C(plan_many_dev, M_i, M_o, CUFFT_INVERSE);
		}
		inline void inverse_trans_many(cufftDoubleComplex*M_i, cufftDoubleComplex*M_o)
		{
			cufftExecZ2Z(plan_many_dev, M_i, M_o, CUFFT_INVERSE);
		}
		cufftHandle plan_dev;
		cufftHandle plan_many_dev;
		int row;
		int col;
		std::size_t workSize;
		std::size_t workSize_many;
		bool float_type;
	};
}