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

	int matrix_size = 4 * sizeof(float);
	float* U = (float*)malloc(matrix_size);

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

	size -= 5;
	int A_size = size * sizeof(float);
	float* A = (float*)malloc(A_size);
	float* B = (float*)malloc(A_size);

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

	cudaError_t err = cudaSuccess;

	float* d_A = NULL;
	err = cudaMalloc((void**)&d_A, A_size);
	if(err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
	}

	float* d_U = NULL;
	err = cudaMalloc((void**)&d_U, matrix_size);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device Matrix U (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
	}

	float* d_B = NULL;
	err = cudaMalloc((void**)&d_B, A_size);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device Output vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_A, A, A_size, cudaMemcpyHostToDevice);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_U, U, matrix_size, cudaMemcpyHostToDevice);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
    	exit(EXIT_FAILURE);
	}

	int threadsPerBlock = 256;
	int blocksPerGrid = (size + threadsPerBlock -1) / threadsPerBlock;

	QubitGate<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_U, d_B, size, q_index);
	
	err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch QubitGate kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	err = cudaMemcpy(B, d_B, A_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy output from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /* Print the output vector */
    for(int i=0 ; i < size; i++) {
    	printf("%.3f\n", B[i]);
    }

    //Free host memory.
    free(A);
    free(B);
    free(U);

    //Free device memory.
    err = cudaFree(d_A);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device memory for input vector (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_U);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device memory for Matrix (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device memory for output vector (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

}