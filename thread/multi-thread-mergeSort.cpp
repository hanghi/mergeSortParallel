#include <iostream>
#include <vector>
#include <cstdlib>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

#define NUM_THREADS 4
int n = 12;

vector<int> generateArray(int n) {
  vector<int> arr;
  srand (time(NULL));

  for(int i = 0; i < n; i++) {
    arr.push_back(rand() % 100);
  }

  srand (1);

  return arr;
}

vector<int> A = generateArray(n);
vector<int> B(n);

typedef struct arg_struct {
  int i;
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

void *BottomUpMerge(void *arg)
{

  //  arg_struct params = *reinterpret_cast<arg_struct*>(arg);
  arg_struct params = *((arg_struct *) arg);

  int iLeft = params.i;
  int iRight = params.min_1;
  int iEnd = params.min_2;

  int i = iLeft;
  int j = iRight;

  // cout << "\nCan go here" << endl;

  // While there are elements in the left or right runs...
  for (int k = iLeft; k < iEnd; k++) {
    // If left run head exists and is <= existing right run head.
    // cout << "i: " << i << endl;
    // cout << "iRight: " << iRight << endl;
    // cout << "j: " << j << endl;
    // cout << "iEnd: " << iEnd << endl;
    // cout << "A[i]: " << A[i] << endl;
    // cout << "A[j]: " << A[j] << endl;

    if (i < iRight && (j >= iEnd || A[i] <= A[j])) {
      B[k] = A[i];
      i = i + 1;
    } else {
      B[k] = A[j];
      j = j + 1;    
    }

    // cout << "Doing sort ..." << endl;
    // Print(B);
    // cout << "End doing " << endl;
  } 

  // cout << "This is A" << endl;
  // Print(A);
  // cout << "iLeft: " << iLeft << endl;
  // cout << "iRight: " << iRight << endl;
  // cout << "iEnd: " << iEnd << endl;
  // cout << "This is B" << endl;
  // Print(B);

  // cout << "===========================" << endl;

  //  sleep(1);

  pthread_exit(NULL);
}

void CopyArray(int n)
{
  for(int i = 0; i < n; i++)
    A[i] = B[i];
}

// array A[] has the items to sort; array B[] is a work array
void BottomUpMergeSort(int n)
{
  int rc;
  int i;


  for (int width = 1; width < n; width = 2 * width)
    {
      //      pthread_t threads[n / width];
      pthread_t threads[n / width];
      pthread_attr_t attr;
      void *status;

      // Initialize and set thread joinable
      pthread_attr_init(&attr);
      pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

      int threadIndex = 0;
      for (int i = 0; i < n; i = i + 2*width)
        {
          arg_struct *arg = new arg_struct;

          arg->i = i;
          arg->min_1 = min(i+width, n);
          arg->min_2 = min(i+2*width, n);

          // cout << "Loop i: " << i << endl;
          // cout << "width " << width << endl;
          // cout << "iLeft " << arg->i << endl;
          // cout << "iRight " << arg->min_1 << endl;
          // cout << "iEnd " << arg->min_2 << endl;

          rc = pthread_create(&threads[threadIndex], &attr, BottomUpMerge, arg);
		
          if (rc){
            cout << "Error:unable to create thread," << rc << endl;
            exit(-1);
          }
          threadIndex += 1;

          //          sleep(1);

        }
      pthread_attr_destroy(&attr);      

      // cout << "what widht: " << width << endl;
      for(int i=0; i < threadIndex; i++ ){
        rc = pthread_join(threads[i], &status);
        if (rc){
          cout << "Error:unable to join," << rc << endl;
          exit(-1);
        }		
      }
      // cout << "All thread done?" << endl;		
      CopyArray(n);
    }
    
}

//  Left run is A[iLeft :iRight-1].
// Right run is A[iRight:iEnd-1  ].



int main() {

  //  vector<int> B(n);

   cout << "Initial Array : ";
   Print(A);

  duration<double> time_span = steady_clock::duration::zero();

  high_resolution_clock::time_point st = high_resolution_clock::now();
  BottomUpMergeSort(n);
  time_span = high_resolution_clock::now() - st;

  cout << "After sorted: ";
  Print(B);
  cout << "Time spent: " << time_span.count() << endl;

  return 0;
}
