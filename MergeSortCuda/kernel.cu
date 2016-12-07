//cuda libs

#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "thrust\merge.h"
//local lib
#include "Helper.h"
#include "MergeSort.h"
#include "MergeSortMultiThreading.h"

//============================MERGE SORT FREE THREADS========================
//Number of threads per block
#define THREADS_PER_BLOCK 512
//compare 2 number, return the minimum
__device__ long min_int(long, long);
//Bottom up merging implementation on gpu device
//Merge 2 parts of array, [l, m) and [m, r)
__device__ void free_merge_bottom_up(int*, int*, long, long, long);
//This function is called by every thread
//each thread will calculate its range from its id, and starting merge on that range
__global__ void gpu_free_thread_merge_sort(int*, int*, long, long);
//Mail call
double gpuMergeSortFreeThreads(int*, long);
//============================================================================
//=============================MERGE SORT FIXED THREADS=======================
//Max thread on single block
#define MAX_THREADS 8
//Copy a part of array B to A
__device__ void fixed_CopyArray(int*, int*, long, long);
//Merge 2 parts of array A (left and right), result save to array B. Split by mid.
__device__ void fixed_BottomUpMerge(int*, int*, long, long , long );
//Array A[] has the items to sort; array B[] is a temporary array
//This function can sort a part of array, start from left and end at right.
__device__ void fixed_BottomUpMergeSort(int*, int* , long, long );
//Thread sort on its part
__global__ void fixed_thread_mergeSort(int* , int* , long );
//Thread merge
__global__ void fixed_thread_merge(int* , int* , long* , long* , long );
//Main call
double gpuMergeSortFixedThreads(int*, long);
//============================MERGESORT USING CUDA TOOLKIT=====================
void thrust_merge(int*a, int left, int mid, int right);
double thrust_BottomUpMergeSort(int* A, long n);
//============================MERGE SORT ON CPU================================
//Merge sort normal
double mergeSortNormal(int*, long);
//Merge sort multi-thread
double mergeSortMT(int*, long);
//-----------------------------------------------------------------------------
//every test case, do 5 times and get average time
void analysis(long length) {

	int n_times = 5;

	double avg_time_1 = 0;
	double avg_time_2 = 0;
	double avg_time_3 = 0;
	double avg_time_4 = 0;
	double avg_time_5 = 0;

	for (int i = 0; i < n_times; i++) {
		//create sample
		int* sample;
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
		//cuda free threads
		avg_time_3 += gpuMergeSortFreeThreads(sample, length);
		//cuda fixed threads
		avg_time_4 += gpuMergeSortFixedThreads(sample, length);
		//cuda toolkit merge
		avg_time_5 += thrust_BottomUpMergeSort(sample, length);
		//
		free(sample);
	}
	avg_time_1 /= n_times;
	avg_time_2 /= n_times;
	avg_time_3 /= n_times;
	avg_time_4 /= n_times;
	avg_time_5 /= n_times;
	//write log
	cout.precision(dbl::max_digits10);
	cout << length << "\t" << fixed << avg_time_1 << "\t" << fixed << avg_time_2 << "\t" << fixed << avg_time_3 << "\t" << fixed << avg_time_4 << "\t" << fixed << avg_time_5 << endl;
	writeLog("result.txt", length, avg_time_1, avg_time_2, avg_time_3, avg_time_4, avg_time_5);
}

void  main() {
	srand(time(NULL));
	//Sample size
	long sample_size = 100;
	//first test case
	analysis(sample_size);
	//19 remaining test case
	for (int i = 1; i < 20; i++) {
		sample_size += (sample_size / 2);//new test case input is larger than old one 50%
		analysis(sample_size);
	}
	cout << endl;
	system("pause");
}
//=============================IMPLEMENTATION=======================
__device__ long min_int(long a, long b) {
	return (a < b) ? a : b;
}

//Merge sort normal
double mergeSortNormal(int* a, long length) {
	//create temp list
	int *temp = (int*)malloc(sizeof(int) * length);
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
double mergeSortMT(int* a, long length) {
	//create temp list
	int *temp = (int*)malloc(sizeof(int) * length);
	//
	high_resolution_clock::time_point watch = high_resolution_clock::now();
	MultiThreadingMergeSort(a, temp, length);
	duration<double> time_span = (high_resolution_clock::now() - watch);
	//free temp
	free(temp);
	//return execution time
	return time_span.count();
}

//====================================CUDA TOOLKIT MERGE SORT==========================
//Merge 2 parts of array using cuda toolkit
void thrust_merge(int*a, int left, int mid, int right) {
	int* temp = new int[right - left];
	thrust::merge(&a[left], &a[left] + (mid - left), &a[mid], &a[mid] + (right - mid), temp);

	//copy to main array
	for (int i = 0; i < right - left; i++)
		a[left + i] = temp[i];

	delete[] temp;
}

//Bottom up merge sort using cuda toolkit
double thrust_BottomUpMergeSort(int* A, long n)
{
	high_resolution_clock::time_point watch = high_resolution_clock::now();
	//
	for (int width = 1; width < n; width = 2 * width)
		for (int i = 0; i < n; i = i + 2 * width)
			thrust_merge(A, i, min(i + width, n), min(i + 2 * width, n));
	//
	duration<double> time_span = (high_resolution_clock::now() - watch);
	return time_span.count();
}

//Copy a part of array B to A
__device__ void fixed_CopyArray(int* B, int* A, long l, long r)
{
	for (int i = l; i < r; i++)
		A[i] = B[i];
}

//Merge 2 parts of array A (left and right), result save to array B. Split by mid.
__device__ void fixed_BottomUpMerge(int* A, int* B, long left, long mid, long right)
{
	int i = left; int j = mid;
	// While there are elements in the left or right runs...
	for (int k = left; k < right; k++) {
		// If left run head exists and is <= existing right run head.
		if (i < mid && (j >= right || A[i] <= A[j])) {
			B[k] = A[i];
			i = i + 1;
		}
		else {
			B[k] = A[j];
			j = j + 1;
		}
	}
}

__device__ void fixed_BottomUpMergeSort(int* A, int* B, long l, long r)
{
	int n = r - l;
	// Each 1-element run in A is already "sorted".
	// Make successively longer sorted runs of length 2, 4, 8, 16... until whole array is sorted.
	for (long width = 1; width < n; width = 2 * width)
	{
		// Array A is full of runs of length width.
		for (long i = l; i < r; i = i + 2 * width)
		{
			// Merge two runs: A[i:i+width-1] and A[i+width:i+2*width-1] to B[]
			// or copy A[i:n-1] to B[] ( if(i+width >= n) )
			fixed_BottomUpMerge(A, B, i, min_int(i + width, r), min_int(i + 2 * width, r));
		}
		// Now work array B is full of runs of length 2*width.
		// Copy array B to array A for next iteration.
		// A more efficient implementation would swap the roles of A and B.
		fixed_CopyArray(B, A, l, r);
		//SwapArray(A, B);
		// Now array A is full of runs of length 2*width.
	}
}

//Thread sort on its part
__global__ void fixed_thread_mergeSort(int* a, int* temp, long n) {
	//Get current ThreadID
	int ThreadID = threadIdx.x;
	//Get the job which is given to thread
	long from = ThreadID *  (n / MAX_THREADS);
	//Sort on that range
	fixed_BottomUpMergeSort(a, temp, from,
		from + ((ThreadID == MAX_THREADS - 1) ? (n / MAX_THREADS) + (n % MAX_THREADS) : (n / MAX_THREADS)));
}

//Thread merge
__global__ void fixed_thread_merge(int* a, int* temp, long* from, long* to, long loc_size) {
	int id = threadIdx.x * 2;

	fixed_BottomUpMerge(a, temp, from[id], to[id], to[id + 1]);
	fixed_CopyArray(temp, a, from[id], to[id + 1]);
	long cur = from[id];
	long cur1 = to[id + 1];
	//
	from[threadIdx.x] = cur;
	to[threadIdx.x] = cur1;
}

double gpuMergeSortFixedThreads(int* a, long n) {
	//device array pointers
	int *dev_working;
	int *dev_temp;
	double exetime = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaSetDevice(0);
	//Allocation gpu memory
	cudaMalloc((void**)&dev_temp, sizeof(long) * n);
	cudaMalloc((void**)&dev_working, sizeof(long) * n);
	//Copy local array to gpu-memory
	cudaMemcpy(dev_working, a, sizeof(long) * n, cudaMemcpyHostToDevice);

	high_resolution_clock::time_point watch = high_resolution_clock::now();
	//threads sort on its range
	fixed_thread_mergeSort << <1, MAX_THREADS >> > (dev_working, dev_temp, n);	
	cudaDeviceSynchronize();
	//
	duration<double> time_span = (high_resolution_clock::now() - watch);
	exetime += time_span.count();

	//threads job simulation
	long* from = (long*)malloc(sizeof(long) * MAX_THREADS);
	long* to = (long*)malloc(sizeof(long) * MAX_THREADS);
	//
	long n_l_jobs = n / MAX_THREADS;
	long n_l_j_remain = n % MAX_THREADS;
	//calculate job and send to gpu, to merge result above
	for (int i = 0; i < MAX_THREADS; i++) {
		long current_jobs = n_l_jobs;
		if (i == MAX_THREADS - 1) //if it's last thread, give it remain job
			current_jobs += n_l_j_remain;

		from[i] = i * n_l_jobs;
		to[i] = i * n_l_jobs + current_jobs;
	}
	//Copy job detail to gpu
	long *dev_from;
	long *dev_to;
	cudaMalloc((void**)&dev_from, sizeof(long) * MAX_THREADS);
	cudaMalloc((void**)&dev_to, sizeof(long) * MAX_THREADS);
	cudaMemcpy(dev_from, from, sizeof(long) * MAX_THREADS, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_to, to, sizeof(long) * MAX_THREADS, cudaMemcpyHostToDevice);

	watch = high_resolution_clock::now();
	//merge results
	for (int range = 1; range < MAX_THREADS; range *= 2)
		fixed_thread_merge <<<1, MAX_THREADS / range >> > (dev_working, dev_temp, dev_from, dev_to, MAX_THREADS);
	//
	cudaDeviceSynchronize();
	//
	time_span = (high_resolution_clock::now() - watch);
	exetime += time_span.count();
	//Copy result from device
	cudaStatus = cudaMemcpy(a, dev_working, sizeof(long) * n, cudaMemcpyDeviceToHost);
	//
	cudaFree(dev_working);	cudaFree(dev_temp); cudaFree(dev_from);	cudaFree(dev_to);
	//return execution time
	return time_span.count();
}

//==================================MERGE SORT FREE THREADS=======================
//Bottom up merging implementation on gpu device
//Merge 2 parts of array, [l, m) and [m, r)
__device__ void free_merge_bottom_up(int* a, int* temp, long l, long m, long r) {
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
__global__ void gpu_free_thread_merge_sort(int* a, int* temp, long n, long width)
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
	free_merge_bottom_up(a, temp, left, mid, right);
}

//This function prepair gpu memory and call kernel function
double gpuMergeSortFreeThreads(int *a, long n) {
	cudaSetDevice(0);
	//device array pointers
	int *dev_working;
	int *dev_temp;

	//Allocation gpu memory
	cudaMalloc((void**)&dev_temp, sizeof(int) * n);
	cudaMalloc((void**)&dev_working, sizeof(int) * n);

	//Copy local array to gpu-memory
	//This line waste a lot of time,
	//algorithm do very quickly but memory copy host-to-device, device-to-host lightly slow.
	cudaMemcpy(dev_working, a, sizeof(long) * n, cudaMemcpyHostToDevice);

	//
	int width;

	//Temporary array
	int* A = dev_working;
	int* B = dev_temp;

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
		gpu_free_thread_merge_sort << <n_blocks, THREADS_PER_BLOCK >> >(A, B, n, width);
		cudaDeviceSynchronize();

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