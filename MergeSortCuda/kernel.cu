//cuda libs
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//local lib
#include "Helper.h"
#include "MergeSort.h"
#include "MergeSortMultiThreading.h"
//Number of threads per block
#define THREADS_PER_BLOCK 512
//compare 2 number, return the minimum
__device__ long min_int(long, long);
//Bottom up merging implementation on gpu device
//Merge 2 parts of array, [l, m) and [m, r)
__device__ void merge_bottom_up(long*, long*, long, long, long);
//This function is called by every thread
//each thread will calculate its range from its id, and starting merge on that range
__global__ void gpu_thread_merge(long*, long*, long, long);
//GPU MERGE SORT ALGORITHM
double gpuMergeSort(long*, long);
//TIME ANALYTICS
//Merge sort normal
double mergeSortNormal(long*, long);
//Merge sort multi-thread
double mergeSortMT(long*, long);

//every test case, do 5 times and get average time
void analysis(int length) {

	int n_times = 1;

	double avg_time_1 = 0;
	double avg_time_2 = 0;
	double avg_time_3 = 0;

	for (int i = 0; i < n_times; i++) {
		//create sample
		long* sample;
		bool g = generateRandomList(length, sample);
		if (!g)
		{
			cout << "Cannot create sample array!\n";
			return;
		}
		//1 thread merge sort	
		avg_time_1 += mergeSortNormal(sample, length);
		//9 threads merge sort
		avg_time_2 += mergeSortMT(sample, length);
		//cuda merge sort
		avg_time_3 += gpuMergeSort(sample, length);
		//
		free(sample);
	}
	avg_time_1 /= n_times;
	avg_time_2 /= n_times;
	avg_time_3 /= n_times;
	//write log
	cout.precision(dbl::max_digits10);
	cout << length << "\t" << fixed << avg_time_1 << "\t" << fixed << avg_time_2 << "\t" << fixed << avg_time_3 << endl;
	writeLog("result.txt", length, avg_time_1, avg_time_2, avg_time_3);
}

void  main() {
	srand(time(NULL));
	//Sample size
	long sample_size = 100;
	//first test case
	analysis(sample_size);
	//22 remaining test case
	for (int i = 1; i < 23; i++) {
		sample_size += (sample_size / 2);//new test case input is larger than old one 50%
		analysis(sample_size);
	}
	cout << endl;
	system("pause");
}

//OTHER MERGE SORT IMPLEMENTATION
//Merge sort normal
double mergeSortNormal(long* a, long length) {
	//create temp list
	long *temp = (long*)malloc(sizeof(long) * length);
	//
	high_resolution_clock::time_point watch = high_resolution_clock::now();
	BottomUpMergeSort(a, temp, 0, length);
	duration<double> time_span = (high_resolution_clock::now() - watch);
	//free temp
	free(temp);
	//return execution time
	return time_span.count();
}

//Merge sort multi-thread
double mergeSortMT(long* a, long length) {
	//create temp list
	long *temp = (long*)malloc(sizeof(long) * length);
	//
	high_resolution_clock::time_point watch = high_resolution_clock::now();
	MultiThreadingMergeSort(a, temp, length);
	duration<double> time_span = (high_resolution_clock::now() - watch);
	//free temp
	free(temp);
	//return execution time
	return time_span.count();
}
//CUDA MERGE SORT - IMPLEMENTATION
//This function prepair gpu memory and call kernel function
double gpuMergeSort(long *a, long n) {

	//device array pointers
	long *dev_working;
	long *dev_temp;

	//Allocation gpu memory
	cudaMalloc((void**)&dev_temp, sizeof(long) * n);
	cudaMalloc((void**)&dev_working, sizeof(long) * n);

	//Copy local array to gpu-memory
	//This line waste a lot of time,
	//algorithm do very quickly but memory copy host-to-device, device-to-host lightly slow.
	cudaMemcpy(dev_working, a, sizeof(long) * n, cudaMemcpyHostToDevice);

	//
	int width;

	//Temporary array
	long* A = dev_working;
	long* B = dev_temp;

	//Clock for watching time
	
	high_resolution_clock::time_point watch = high_resolution_clock::now();
	cudaError_t err;
	//Split array to ranges, each range has length equal to width
	//width is multiplied by 2
	for (width = 1; width < n; width *= 2) {
		//number of threads need to use
		long n_threads_need = n / width;
		//number of blocks from n_threads_need
		long n_blocks = (n_threads_need + (THREADS_PER_BLOCK - 1)) / (THREADS_PER_BLOCK);
		//call kernel
		gpu_thread_merge <<<n_blocks, THREADS_PER_BLOCK >>>(A, B, n, width);
		cudaDeviceSynchronize();

		err = cudaGetLastError();
		if (err != cudaSuccess)
			cout << n << " meet e: " << cudaGetErrorString(err) << endl;

		//swap array
		A = A == dev_working ? dev_temp : dev_working;
		B = B == dev_working ? dev_temp : dev_working;
	}
	//stop clock
	duration<double> time_span = (high_resolution_clock::now() - watch);

	//Copy result to local memory
	//This line waste alot of time,
	//algorithm is very quick but memory copy host-to-device, device-to-host lightly slow.
	cudaMemcpy(a, A, sizeof(long) * n, cudaMemcpyDeviceToHost);
	//Free gpu memory
	cudaFree(dev_temp);
	cudaFree(dev_working);
	//Return execution time of merge sort
	//
	return time_span.count();
}

//Bottom up merging implementation on gpu device
//Merge 2 parts of array, [l, m) and [m, r)
__device__ void merge_bottom_up(long* a, long* temp, long l, long m, long r) {
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

//This function is called by every thread
//each thread will calculate its range from its id, and starting merge on that range
__global__ void gpu_thread_merge(long* a, long* temp, long n, long width)
{
	//id of thread
	long idx = blockIdx.x * blockDim.x + threadIdx.x;
	//define working range of thread
	//range begining
	long i = idx * 2 * width;
	if (i >= n)
		return;
	//range detail
	long left = i;
	long mid = min_int(n, i + width);
	long right = min_int(n, i + 2 * width);
	//Do bottom up merge
	merge_bottom_up(a, temp, left, mid, right);
}

//Compare 2 number, return the minimum
//This funcion running on gpu device
__device__ long min_int(long a, long b) {
	return (a < b) ? a : b;
}
