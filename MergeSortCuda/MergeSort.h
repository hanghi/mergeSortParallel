#pragma once
#define min(a, b)(a < b ? a : b)
//--------------------------Implementation
//Copy a part of array B to A
void CopyArray(long*, long*, int, int);
//Swap 2 pointers
void SwapArray(long*, long*);
//Merge 2 parts of array A (left and right), result save to array B. Split by mid.
void BottomUpMerge(long*, long*, int, int, int);
//Array A[] has the items to sort; array B[] is a temporary array
//This function can sort a part of array, start from left and end at right.
void BottomUpMergeSort(long*, long*, int, int);
