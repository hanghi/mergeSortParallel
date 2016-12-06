#pragma once
#include <thread>
#include <vector>
#include "MergeSort.h"
using namespace std;
//
//Number of threads
#define THREADS 8
//Job struct for each thread
struct JOB {
	long from;
	long to;
	JOB(long _from, long _to) {
		from = _from;
		to = _to;
	}
};
//List of structures with sort name
typedef vector<thread> T;
typedef vector<JOB> Jobs;
//---------------------------------------Implementation
//Merge result from threads
Jobs mergeSubArray(Jobs, long*, long*);
//Multi threads merge sort
void MultiThreadingMergeSort(long*, long*, long);