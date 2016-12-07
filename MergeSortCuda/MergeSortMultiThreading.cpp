#include "MergeSortMultiThreading.h"

//Merge result from threads
Jobs mergeSubArray(Jobs input, int* a, int* temp) {
	Jobs ret;
	//Each round merge 2 job
	//Merge 0 vs 1, 2 vs 3, 4 vs 5, ....
	for (int i = 0; i < input.size(); i += 2) {
		//Bottom up merge
		BottomUpMerge(a, temp, input[i].from, input[i].to, input[i + 1].to);
		//Swap temp to origin
		//SwapArray(a, temp);
		CopyArray(temp, a, input[i].from, input[i + 1].to);
		//add merging's result to vector
		ret.push_back(JOB(input[i].from, input[i + 1].to));
	}
	//return result
	return ret;
}

//Multi threads merge sort
void MultiThreadingMergeSort(int* a, int* temp, long n) {
	//Get length of job which is given to thread
	long n_l_jobs = n / THREADS;
	//Last thread will do this remain job
	long n_l_j_remain = n % THREADS;
	//List of jobs
	Jobs l_jobs;
	//List of threads
	T ThreadList;

	//Delivery jobs to threads	
	for (int i = 0; i < THREADS; i++) {
		//get current job
		long current_jobs = n_l_jobs;
		if (i == THREADS - 1) //if it's last thread, give it remain job
			current_jobs += n_l_j_remain;

		//where job is on array, calculate range of job
		long from = i * n_l_jobs;
		long to = from + current_jobs;
		//New thread will sort on that range
		thread t(BottomUpMergeSort, a, temp, from, to);
		//Add to thread list
		ThreadList.push_back(move(t));
		//Add current job to list, to prepair merging when thread done
		l_jobs.push_back(JOB(from, to));
	}

	//Wait all threads
	for (T::iterator it = ThreadList.begin(); it != ThreadList.end(); ++it) {
		if (it->joinable())
			it->join();
	}

	//Merge thread's results
	Jobs tempJobs;
	for (int range = 1; range < THREADS; range *= 2)
		if (range == 1)
			tempJobs = mergeSubArray(l_jobs, a, temp);
		else
			tempJobs = mergeSubArray(tempJobs, a, temp);
	//Free memory
	l_jobs.clear();
	tempJobs.clear();
}
