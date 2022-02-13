#include <stdio.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <sys/time.h>


using namespace std;

__global__ void QubitGate (float* A, float* U, float* B, int size, int q_index) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = i/q_index; //Gives the quotient.

	if((i < size) && ((i+q_index) < size) && (j %2 == 0)) {
			B[i] = A[i]*U[0] + A[i + q_index]* U[1];
			B[i + q_index] = A[i]*U[2] + A[i + q_index]*U[3];
	}
}


int main (int argc, char* argv[]) {
	ifstream file;
	char* input_file;

	int q_index; 
	int size = 0;

	if(argc != 2) {
		printf("Error: Expected File Name\n");
		exit(EXIT_FAILURE);
	}
	input_file = argv[1];

	string s;
	file.open(input_file);
	if(file.is_open()) {
		while(!file.eof()) {
			file >> s;
			size++;
		}
	}
	else {
		printf("Error: Can't Open File\n");
		exit(EXIT_FAILURE);
	}
	file.close();

	cudaError_t err = cudaSuccess;
	size -= 5;
	int A_size = size * sizeof(float);
	
	float* A = NULL;
	err = cudaMallocManaged(&A, A_size);
	if(err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate vector A in Unified Memory (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
	}

	float* B = NULL;
	err = cudaMallocManaged(&B, A_size);
	if(err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate output vector B in Unified Memory (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
	}

	int matrix_size = 4 * sizeof(float);
	float* U = NULL;
	err = cudaMallocManaged(&U, matrix_size);
	if(err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate matrix U in Unified Memory (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
	}

	file.open(input_file);
	if(file.is_open()) {
		file >> U[0] >> U[1] >> U[2] >> U[3];

		for(int i=0; i < size; i++) {
			file >> A[i];
		}

		file >> q_index;
	}
	else {
		printf("Error: Can't Open File again\n");
		exit(EXIT_FAILURE);
	}

	q_index = 1 << q_index;

	int threadsPerBlock = 256;
	int blocksPerGrid = (size + threadsPerBlock -1) / threadsPerBlock;

	QubitGate<<<blocksPerGrid, threadsPerBlock>>>(A, U, B, size, q_index);

	err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch QubitGate kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to Synchronize (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Print the output vector.
    for(int i=0 ; i < size; i++) {
    	printf("%.3f\n", B[i]);
    }
    
    //Free memory.
    err = cudaFree(A);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free Unified Memory for Input Vector (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(B);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free Unified Memory for Matrix (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(U);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free Unified Memory for Output Vector (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

}