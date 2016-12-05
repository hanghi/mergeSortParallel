#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

void Print (const vector<int>& v){
  //vector<int> v;
  for (int i=0; i<v.size();i++){
    cout << v[i] << ' ';
  }
  cout << endl;
}

vector<int> generateArray(int n) {
  vector<int> arr;
  srand (time(NULL));

  for(int i = 0; i < n; i++) {
    arr.push_back(rand() % 1000);
  }

  srand (1);

  return arr;
}

void BottomUpMerge(vector<int> &A, int iLeft, int iRight, int iEnd, vector<int> &B)
{


  int i = iLeft;
  int j = iRight;
    // While there are elements in the left or right runs...
    for (int k = iLeft; k < iEnd; k++) {
        // If left run head exists and is <= existing right run head.
        if (i < iRight && (j >= iEnd || A[i] <= A[j])) {
            B[k] = A[i];
            i = i + 1;
        } else {
            B[k] = A[j];
            j = j + 1;    
        }
    } 
}

void CopyArray(vector<int> &B, vector<int> &A, int n)
{
    for(int i = 0; i < n; i++)
        A[i] = B[i];
}

// array A[] has the items to sort; array B[] is a work array
void BottomUpMergeSort(vector<int> &A, vector<int> &B, int n)
{
    for (int width = 1; width < n; width = 2 * width)
    {
        for (int i = 0; i < n; i = i + 2 * width)
        {
            BottomUpMerge(A, i, min(i+width, n), min(i+2*width, n), B);
        }
        CopyArray(B, A, n);
    }
    
}

//  Left run is A[iLeft :iRight-1].
// Right run is A[iRight:iEnd-1  ].


int main() {
  int n = 1000000;

  vector<int> A = generateArray(n);
  vector<int> B(n);

  cout << "Intial Array : ";
  //  Print(A);

  duration<double> time_span = steady_clock::duration::zero();

  high_resolution_clock::time_point st = high_resolution_clock::now();
  BottomUpMergeSort(A, B, n);
  time_span = high_resolution_clock::now() - st;

  cout << "After sorted: ";
  //  Print(B);
  cout << "Time spent: " << time_span.count() << endl;

  return 0;
}
