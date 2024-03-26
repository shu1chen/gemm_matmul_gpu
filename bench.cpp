#include <array>
#include <vector>

#include <iostream>
#include <chrono>
#include <unordered_map>

#include <fstream>

#include <sycl/sycl.hpp>
#include <mkl.h>
#include "oneapi/mkl/blas.hpp"
#include "oneapi/dnnl/dnnl.hpp"

using namespace dnnl;
using tag = memory::format_tag;
using dt = memory::data_type;

//
// Random initialization of scalar, vector, general matrix and triangular matrix
//
template <typename fp> fp rand_scalar() { return fp(std::rand()) / fp(RAND_MAX) - fp(0.5); }
template <typename fp> fp rand_scalar(int mag) { fp tmp = fp(mag) + fp(std::rand()) / fp(RAND_MAX) - fp(0.5); if (std::rand() % 2) return tmp; else return -tmp; }

template <typename fp> void rand_matrix(fp *M, oneapi::mkl::transpose trans, int m, int n, int ld)
{
    if (trans == oneapi::mkl::transpose::nontrans) {
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++)
                M[i + j * ld] = rand_scalar<fp>();
    } else {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                M[j + i * ld] = rand_scalar<fp>();
    }
}

void printDNNLStatus(dnnl_status_t &status)
{
	if (status == dnnl_success)
	{
		std::cout << "DNNL success." << std::endl;
	}
	else if (status == dnnl_out_of_memory)
	{
		std::cout << "The operation failed due to an out-of-memory condition." << std::endl;
	}
	else if (status == dnnl_invalid_arguments)
	{
		std::cout << "The operation failed because of incorrect function arguments." << std::endl;
	}
	else if (status == dnnl_unimplemented)
	{
		std::cout << "The operation failed because requested functionality is not implemented." << std::endl;
	}
	else if (status == dnnl_last_impl_reached)
	{
		std::cout << "Primitive iterator passed over last primitive descriptor." << std::endl;
	}
	else
	{
		std::cout << "onednn error: " << status << std::endl;
	}
}

struct matrix_size
{
	const int M;
	const int K;
	const int N;

	friend std::ostream &operator<<(std::ostream &os, const matrix_size &m)
	{
		os << "Matrix size: M: " << m.M << " K: " << m.K << " N: " << m.N;
		return os;
	}
};

std::string generateTimestamp()
{
	std::time_t now = std::time(nullptr);
	char timestamp[20];
	std::strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", std::localtime(&now));
	return std::string(timestamp);
}

template <typename fp>
void benchmarkLoop(int iterations, std::vector<matrix_size> &matrices, const sycl::device &dev)
{
	std::chrono::duration<double> dnnl_matmul_duration_loop = std::chrono::duration<double>::zero();
	std::chrono::duration<double> mkl_cblas_sgemm_duration_loop = std::chrono::duration<double>::zero();

	std::string timestamp = generateTimestamp();
	std::string filename = "gpu_perf_data_" + timestamp + ".csv";
	std::ofstream outFile(filename, std::ios::out);
	if (!outFile.is_open())
	{
		std::cerr << "Error: opening file failed." << std::endl;
		return;
	}
	outFile << "Arch,Matrix size MxKxN,Iterations,DNNL matmul avg (ms),MKL gemm avg (ms),XETLA gemm avg (ms)" << std::endl;

	for (auto &&sizes : matrices)
	{
		oneapi::mkl::transpose transA = oneapi::mkl::transpose::nontrans;
		oneapi::mkl::transpose transB = oneapi::mkl::transpose::nontrans;
		oneapi::mkl::transpose transC = oneapi::mkl::transpose::nontrans;
		const int M = sizes.M;
		const int K = sizes.K;
		const int N = sizes.N;
		fp alpha = 1;
		fp beta = 1;
		int lda = K;
		int ldb = N;
		int ldc = N;

		// Catch asynchronous exceptions
		auto exception_handler = [] (sycl::exception_list exceptions) {
			for (std::exception_ptr const& e : exceptions) {
				try {
					std::rethrow_exception(e);
				} catch(sycl::exception const& e) {
					std::cout << "Caught asynchronous SYCL exception during GEMM:\n"
					<< e.what() << std::endl;
				}
			}
		};

		// create execution queue and buffers of matrix data
		sycl::queue queue(dev, exception_handler);
		sycl::event gemm_done;
		std::vector<sycl::event> gemm_dependencies;
		sycl::context cxt = queue.get_context();

		// https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2024-0/gemm.html
		// Row major
		int sizea = ldA * M;
		int sizeb = ldB * K;
		int sizec = ldC * M;
		auto A = (fp *)malloc_shared(sizea * sizeof(fp), dev, cxt);
		auto B = (fp *)malloc_shared(sizeb * sizeof(fp), dev, cxt);
		auto C = (fp *)malloc_shared(sizec * sizeof(fp), dev, cxt);

		//// DNNL Matmul
		// https://oneapi-src.github.io/oneDNN/dev_guide_dpcpp_interoperability.html
		// Create execution dnnl::engine.
		dnnl::engine engine = dnnl::sycl_interop::make_engine(dev, cxt);
		// Create dnnl::stream.
		dnnl::stream strm(engine);
		// Source (A), weights (B), and destination (C) matrix dimensions.
		memory::dims a_dims = {M, K};
		memory::dims b_dims = {K, N};
		memory::dims c_dims = {M, N};
		memory::data_type type = fp;
		// Create memory descriptors and memory objects for src, weights, bias, and dst.
		auto a_md = memory::desc(a_dims, type, tag::any);
		auto b_md = memory::desc(b_dims, type, tag::any);
		auto c_md = memory::desc(c_dims, type, tag::any);
		auto a_in_md = memory::desc(a_dims, type, tag::ab);
		auto b_in_md = memory::desc(b_dims, type, tag::ab);

		auto a_in_mem = sycl_interop::make_memory(
            a_in_md, engine, sycl_interop::memory_kind::usm, A);
		auto b_in_mem = sycl_interop::make_memory(
            b_in_md, engine, sycl_interop::memory_kind::usm, B);

		// Create primitive descriptor.
		auto matmul_pd = matmul::primitive_desc(engine, a_md, b_md, c_md);
		// Repack and convert input data.
		auto a_mem = memory(matmul_pd.src_desc(), engine);
		reorder(a_in_mem, a_mem).execute(strm, a_in_mem, a_mem);
		auto b_mem = memory(matmul_pd.weights_desc(), engine);
		reorder(b_in_mem, b_mem).execute(strm, b_in_mem, b_mem);
		auto c_mem = memory(matmul_pd.dst_desc(), engine);
		// Create the primitive.
		auto matmul_prim = matmul(matmul_pd);
		// Primitive arguments.
		std::unordered_map<int, memory> matmul_args;
		matmul_args.insert({DNNL_ARG_SRC, a_mem});
		matmul_args.insert({DNNL_ARG_WEIGHTS, b_mem});
		matmul_args.insert({DNNL_ARG_DST, c_mem});

		for (int i = 0; i < iterations + 1; i++)
		{
			// Write data to memory object's handles.
			rand_matrix(A, transA, M, K, ldA);
			rand_matrix(B, transB, K, N, ldB);
			rand_matrix(C, transC, M, N, ldC);

			{ // MKL cblas_sgemm
				auto mkl_start = std::chrono::system_clock::now();

				// add oneapi::mkl::blas::gemm to execution queue
				try {
					gemm_done = oneapi::mkl::blas::gemm(queue, transA, transB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC, gemm_dependencies);
				}
				catch(sycl::exception const& e) {
					std::cout << "\t\tCaught synchronous SYCL exception during GEMM:\n"
							<< e.what() << std::endl << "OpenCL status: " << get_error_code(e) << std::endl;
				}

				gemm_done.wait();

				mkl_cblas_sgemm_duration_loop += (mkl_end - mkl_start);
			}

			// DNNL matmul
			{
				auto dnnl_matmul_start = std::chrono::system_clock::now();

				matmul_prim.execute(strm, matmul_args);
				strm.wait();

				auto dnnl_matmul_end = std::chrono::system_clock::now();

				dnnl_matmul_duration_loop += (dnnl_matmul_end - dnnl_matmul_start);
			}

			/*First dnnl and fbgemm calls are slow, so ignore results from the first run of the loop*/
			if (i == 0)
			{
				dnnl_matmul_duration_loop = std::chrono::duration<double>::zero();
				mkl_cblas_sgemm_duration_loop = std::chrono::duration<double>::zero();
			}
		}
		std::cout << std::fixed;
		std::cout.precision(3);
		std::cout << sizes << " in loop, for " << iterations << " iterations, avg:" << std::endl;

		std::cout << "               DNNL matmul took: " << dnnl_matmul_duration_loop.count() * 10e6 / iterations << " ms." << std::endl;
		std::cout << "                  MKL gemm took: " << mkl_cblas_sgemm_duration_loop.count() * 10e6 / iterations << " ms." << std::endl;

		outFile << M << "x" << K << "x" << N << "," << iterations << ","
				<< dnnl_matmul_duration_loop.count() * 10e6 / iterations << ","
				<< mkl_cblas_sgemm_duration_loop.count() * 10e6 / iterations << ","
				<< std::endl;

		mkl_free_buffers();
	}

	outFile.close();
	std::cout << "Data has been written to " << filename << "." << std::endl;
}

int main(int argc, char const *argv[])
{
	int iterations = 100;
	if (argc == 1)
	{
		iterations = 100;
	}
	else if (argc == 2)
	{
		iterations = std::atoi(argv[1]);
	}
	else
	{
		std::cerr << "Usage: " << argv[0] << " [iterations=100] " << std::endl;
		std::exit(1);
	}

	std::vector<matrix_size> matrices = {
		{1024, 1024, 1024},
		{256, 10368, 256},
		{256, 5312, 256},
		{8, 2048, 256},
		{320, 256, 256},
		{472, 256, 256},
		{248, 256, 256},
		{200, 256, 256},
		{1, 64, 8}};

	sycl::device my_dev = sycl::device(sycl::gpu_selector_v);

	std::cout << "On GPU. Running with single precision real data type:" << std::endl;
	benchmarkLoop<float>(iterations, matrices, my_dev);
	if (my_dev.get_info<sycl::info::device::double_fp_config>().size() != 0) {
	    std::cout << "On GPU. Running with double precision real data type:" << std::endl;
	    benchmarkLoop<double>(iterations, matrices, my_dev);
	}

	return 0;
}
