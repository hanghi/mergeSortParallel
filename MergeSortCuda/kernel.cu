//cuda libs
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//windows libs
#include <iostream>
#include <time.h>
#include <chrono>
#include <limits>
using namespace std;
using namespace std::chrono;
typedef std::numeric_limits< double > dbl;
//define threads and block
#define THREADS_PER_BLOCK 512
//compare 2 number, return the minimum
__device__ int min_int(int a, int b) {
	return (a < b) ? a : b;
}

__device__ void merge_bottom_up(int* a, int* temp, int l, int m, int r) {
	int i = l, j = m;
	// While there are elements in the left or right runs...
	for (int k = l; k < r; k++) {
		// If left run head exists and is <= existing right run head.
		if (i < m && (j >= r || a[i] <= a[j])) {
			temp[k] = a[i];
			i = i + 1;
		}
		else {
			temp[k] = a[j];
			j = j + 1;
		}
	}
}

__global__ void gpu_thread_merge(int* a, int* temp, int n, int width)
{
	//id of thread
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//define working range of thread
	//range begining
	int i = idx * 2 * width;
	if (i >= n)
		return;
	//range detail
	int left = i;
	int mid = min_int(n, i + width);
	int right = min_int(n, i + 2 * width);
	//Do bottom up merge
	merge_bottom_up(a, temp, left, mid, right);
}

double cxtime;

void mergeSort(int *a, int n) {

	//slice array to pieces and each has own width
	int *dev_temp;
	int *dev_working;
	//prepare memory
	cudaMalloc((void**)&dev_temp, sizeof(int) * n);
	cudaMalloc((void**)&dev_working, sizeof(int) * n);
	//copy array to gpu-memory
	cudaMemcpy(dev_working, a, sizeof(int) * n, cudaMemcpyHostToDevice);
	//
	int width;
	//
	int* tmp_T = dev_temp;
	int* tmp_W = dev_working;

	//merge
	high_resolution_clock::time_point watch = high_resolution_clock::now();
	//
	for (width = 1; width < n; width *= 2) {
		//number of threads need to use
		int n_threads_need = n / width;
		//number of blocks from n_threads_need
		int n_blocks = (n_threads_need + (THREADS_PER_BLOCK - 1)) / (THREADS_PER_BLOCK);
		//call kernel
		gpu_thread_merge<<<n_blocks, THREADS_PER_BLOCK>>>(tmp_W, tmp_T, n, width);
		//swap
		tmp_W = tmp_W == dev_working ? dev_temp : dev_working;
		tmp_T = tmp_T == dev_working ? dev_temp : dev_working;
	}

	duration<double> time_span = (high_resolution_clock::now() - watch);
	cxtime = time_span.count();
	//free(a);
	//(int*)a =(int*) malloc(sizeof(int) * n);
	cudaMemcpy(a, dev_working, sizeof(int) * n, cudaMemcpyDeviceToHost);
	//
	cudaFree(dev_temp);
	cudaFree(dev_working);
}

//Generate sorted list with length given
int* generateRandomList(int length) {
	//
	srand(time(NULL));
	int* list = (int*) malloc(sizeof(int)*length);

	for (int i = 0; i < length; i++)	
		list[i] = rand() % length;
	
	return list;
}

void showArray(int *a, int l) {
	for (int i = 0; i < l; i++) {
		cout << a[i] << "  ";
	}
	cout << endl;
}

int main()
{
	int n = 3238665;
	cout << "generate list" << endl;
	cout << "length = " << n << endl;
	int* list = generateRandomList(n);
	cout << "start algorithm" << endl;
	mergeSort(list, n);
	//Measure time
	cout.precision(dbl::max_digits10);
	cout << "time: " << "\t" << fixed << cxtime << endl;
	cout << endl;
	system("pause");
    return 0;
}
