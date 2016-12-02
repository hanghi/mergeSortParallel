#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

#define NUM_THREADS 5

typedef struct arg_struct {
  int i;
  vector<int> A;
  vector<int> B;
  int min_1;
  int min_2;
} arg_struct;

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


void *BottomUpMerge(void *arg)
{
  arg_struct *params = arg;  

  vector<int> A = params->A;
  vector<int> B = params->B;

  int iLeft = params->i;
  int iRight = params->min_1;
  int iEnd = params->min_2;

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
  int rc;
  int i;

  for (int width = 1; width < n; width = 2 * width)
    {
      pthread_t threads[n / width];
      pthread_attr_t attr;
      void *status;

      // Initialize and set thread joinable
      pthread_attr_init(&attr);
      pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

      int threadIndex = 0;
      for (int i = 0; i < n; i = i + 2 * width)
        {
          arg_struct arg;
          arg.A = A;
          arg.B = B;
          arg.i = i;
          arg.min_1 = min(i+width, n);
          arg.min_2 = min(i+2*width, n);
          rc = pthread_create(&threads[threadIndex], &attr, BottomUpMerge, A, &arg);
		
          if (rc){
            cout << "Error:unable to create thread," << rc << endl;
            exit(-1);
          }
          threadIndex += 1;
        }

      pthread_attr_destroy(&attr);      
      for(int i=0; i < n/width; i++ ){
        rc = pthread_join(threads[i], &status);
		
        if (rc){
          cout << "Error:unable to join," << rc << endl;
          exit(-1);
        }		
      }

      CopyArray(B, A, n);
    }
    
}

//  Left run is A[iLeft :iRight-1].
// Right run is A[iRight:iEnd-1  ].


int main() {

  int n = 100000000;

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
