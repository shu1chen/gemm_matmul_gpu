#include <array>
#include <vector>

#include <iostream>
#include "oneapi/dnnl/dnnl.hpp"
#include <chrono>

#include <unordered_map>

#include <fstream>

#ifdef WITH_MKL
#include <mkl.h>
#endif

using namespace dnnl;
using tag = memory::format_tag;
using dt = memory::data_type;

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

void benchmarkLoop(int iterations, std::vector<matrix_size> &matrices, const size_t align)
{
#ifdef WITH_MKL
	if (myarch.mkl_ >= 0)
		mkl_enable_instructions(myarch.mkl_);
#endif

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

		char offsetc = 'F';
		bool zero_oa = 1;
		bool zero_ob = 1;
		bool zero_oc = 0;
		char transA = 'N';
		char transB = 'N';
		const int M = sizes.M;
		const int K = sizes.K;
		const int N = sizes.N;
		float alpha = 1;
		float beta = 1;
		int lda = K;
		int ldb = N;
		int ldc = N;
		int8_t oa = 0;
		int8_t ob = 0;
		std::array<int32_t, 1> oc = {0};

		//// DNNL Matmul
		// Create execution dnnl::engine.
		dnnl::engine engine(engine::kind::gpu, 0);
		// Create dnnl::stream.
		dnnl::stream engine_stream(engine);
		// Source (A), weights (B), and destination (C) matrix dimensions.
		memory::dims a_dims = {M, K};
		memory::dims b_dims = {K, N};
		memory::dims c_dims = {M, N};
		memory::data_type type = dt::f32;
		// Create memory descriptors and memory objects for src, weights, bias, and dst.
		auto a_md = memory::desc(a_dims, type, tag::any);
		auto b_md = memory::desc(b_dims, type, tag::any);
		auto c_md = memory::desc(c_dims, type, tag::any);
		auto a_in_md = memory::desc(a_dims, type, tag::ab);
		auto b_in_md = memory::desc(b_dims, type, tag::ab);
		auto a_in_mem = memory(a_in_md, engine);
		auto b_in_mem = memory(b_in_md, engine);
		// Create primitive descriptor.
		auto matmul_pd = matmul::primitive_desc(engine, a_md, b_md, c_md);
		// Repack and convert input data.
		auto a_mem = memory(matmul_pd.src_desc(), engine);
		reorder(a_in_mem, a_mem).execute(engine_stream, a_in_mem, a_mem);
		auto b_mem = memory(matmul_pd.weights_desc(), engine);
		reorder(b_in_mem, b_mem).execute(engine_stream, b_in_mem, b_mem);
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
#ifdef WITH_MKL
			if (use_fp32)
			{ // MKL cblas_sgemm
				alloc::AlignedVector<float> A_MKL(M * K, align);
				alloc::AlignedVector<float> B_MKL(K * N, align);
				alloc::AlignedVector<float> C_MKL(M * N, align);

				std::copy(kenneth_a_tmp.data(), kenneth_a_tmp.data() + kenneth_a_tmp.size(), A_MKL.get());
				std::copy(kenneth_b_tmp.data(), kenneth_b_tmp.data() + kenneth_b_tmp.size(), B_MKL.get());
				std::copy(C.data(), C.data() + C.size(), C_MKL.get());

				auto mkl_start = std::chrono::system_clock::now();
				cblas_sgemm(CblasRowMajor,
							transA == 'N' ? CblasNoTrans : CblasTrans,
							transB == 'N' ? CblasNoTrans : CblasTrans,
							/*CblasFixOffset,*/
							M, N, K,
							alpha,
							A_MKL.get(), lda, // oa,
							B_MKL.get(), ldb, // ob,
							beta,
							C_MKL.get(), ldc); // oc.data());
				auto mkl_end = std::chrono::system_clock::now();

				mkl_cblas_sgemm_duration_loop += (mkl_end - mkl_start);
			}
#endif

			// DNNL matmul
			{
				alloc::AlignedVector<float> A_DNNL_MATMUL(M * K, align);
				alloc::AlignedVector<float> B_DNNL_MATMUL(K * N, align);
				alloc::AlignedVector<float> C_DNNL_MATMUL(M * N, align);

				// Write data to memory object's handles.
				std::copy(kenneth_a_tmp.data(), kenneth_a_tmp.data() + kenneth_a_tmp.size(), static_cast<uint8_t *>(a_in_mem.get_data_handle()));
				std::copy(kenneth_b_tmp.data(), kenneth_b_tmp.data() + kenneth_b_tmp.size(), static_cast<uint8_t *>(b_in_mem.get_data_handle()));

				auto dnnl_matmul_start = std::chrono::system_clock::now();

				matmul_prim.execute(engine_stream, matmul_args);
				engine_stream.wait();

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
#ifdef WITH_MKL
		if (use_fp32)
			std::cout << "                  MKL gemm took: " << mkl_cblas_sgemm_duration_loop.count() * 10e6 / iterations << " ms." << std::endl;
#endif

		std::cout << "Alignment was: " << align << "." << std::endl;

		outFile << M << "x" << K << "x" << N << "," << iterations << ","
				<< dnnl_matmul_duration_loop.count() * 10e6 / iterations << ","
				<< mkl_cblas_sgemm_duration_loop.count() * 10e6 / iterations << ","
				<< std::endl;
	}

	outFile.close();
	std::cout << "Data has been written to " << filename << "." << std::endl;
}

int main(int argc, char const *argv[])
{

	// auto status = dnnl_set_max_cpu_isa(dnnl_cpu_isa_avx512_core);

	size_t align = 64;

	int iterations = 100;
	if (argc == 1)
	{
		iterations = 100;
		align = 64;
	}
	else if (argc == 2)
	{
		iterations = std::atoi(argv[1]);
	}
	else if (argc == 3)
	{
		iterations = std::atoi(argv[1]);
		align = std::atoi(argv[2]);
	}
	else
	{
		std::cerr << "Usage: " << argv[0] << " [iterations=100] [align=64]" << std::endl;
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

	benchmarkLoop(iterations, matrices, align);

	return 0;
}
