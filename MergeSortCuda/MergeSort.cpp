#include "MergeSort.h"
//Copy a part of array B to A
void CopyArray(long* B, long* A, int l, int r)
{
	for (int i = l; i < r; i++)
		A[i] = B[i];
}

/////this implementation was failure :((
void SwapArray(long* A, long* B) {
	long* tempA = A;
	long* tempB = B;
	//
	long* swap = tempA;
	tempA = tempB;
	tempB = swap;
}

//Merge 2 parts of array A (left and right), result save to array B. Split by mid.
void BottomUpMerge(long* A, long* B, int left, int mid, int right)
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

//Array A[] has the items to sort; array B[] is a temporary array
//This function can sort a part of array, start from left and end at right.
void BottomUpMergeSort(long* A, long* B, int l, int r)
{
	int n = r - l;
	// Each 1-element run in A is already "sorted".
	// Make successively longer sorted runs of length 2, 4, 8, 16... until whole array is sorted.
	for (int width = 1; width < n; width = 2 * width)
	{
		// Array A is full of runs of length width.
		for (int i = l; i < r; i = i + 2 * width)
		{
			// Merge two runs: A[i:i+width-1] and A[i+width:i+2*width-1] to B[]
			// or copy A[i:n-1] to B[] ( if(i+width >= n) )
			BottomUpMerge(A, B, i, min(i + width, r), min(i + 2 * width, r));
		}
		// Now work array B is full of runs of length 2*width.
		// Copy array B to array A for next iteration.
		// A more efficient implementation would swap the roles of A and B.
		CopyArray(B, A, l, r);
		//tA = tA == A ? A : B;
		//tB = tB == A ? A : B;
		//SwapArray(A, B);
		// Now array A is full of runs of length 2*width.
	}
}
