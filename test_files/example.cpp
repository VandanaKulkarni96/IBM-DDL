#include <ddl.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <iostream>

#define N 100

int main(int argc, char* argv[]) {
	//Obtain options for DDL from the DDL_OPTIONS environment variable
	std::string options = getenv("DDL_OPTIONS");

	//Initialize DDL
	ddl_init(options.c_str());

	// Determine rank of process
	int rank;
	ddl_rank(&rank);

	//Allocate memory on CPU
	float* cpu_buffer = new float[N];
	float* gpu_buffer;

	//Initialize buffer with data
	 for (int i = 0; i < N; i++)
        cpu_buffer[i] = (i % 100);

	//Allocate memory on the GPU
	int ngrad = ddl_malloc((void**)& gpu_buffer, 64, N * sizeof(float)) /
	sizeof(float);
    
	//Copy buffer from the CPU to the GPU
	cudaMemcpy(gpu_buffer, cpu_buffer, N * sizeof(float), cudaMemcpyHostToDevice);

	// Perform DDL's allreduce function
	ddl_allreduce(gpu_buffer, ngrad);
	
	// Copy buffer from the GPU to the CPU
	cudaMemcpy(cpu_buffer, gpu_buffer, N * sizeof(float), cudaMemcpyDeviceToHost);
	//
	// Print out buffer on a single node:
	if (rank == 0) {
		for (int i = 0; i < N; i++)
			std::cout << cpu_buffer[i] << " ";                                                 	   std::cout << std::endl;
	}

	//Finalize DDL
        ddl_finalize();
        return 0;
}
