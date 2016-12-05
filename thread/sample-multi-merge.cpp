#include <stdio.h>
#include <vector>
#include <stdlib.h>
#include <pthread.h>
#include <iostream>
using namespace std;

#define N 2  /* # of thread */

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

vector<int> a = generateArray(15);  /* target array */
//vector<int> a = {3,5,1};
//int a[] = {15, 19, 1, 93, 6, 5, 12, 3, 2, 10};

/* structure for array index
 * used to keep low/high end of sub arrays
 */
typedef struct Arr {
    int low;
    int high;
} ArrayIndex;

void merge(int low, int high)
{
        int mid = (low+high)/2;
        int left = low;
        int right = mid+1;

        int b[high-low+1];
        int i, cur = 0;

        while(left <= mid && right <= high) {
                if (a[left] > a[right])
                        b[cur++] = a[right++];
                else
                        b[cur++] = a[left++];
        }

        while(left <= mid) b[cur++] = a[left++];
        while(right <= high) b[cur++] = a[right++];
        for (i = 0; i < (high-low+1) ; i++) a[low+i] = b[i];
}

void * mergesort(void *a)
{
        ArrayIndex *pa = (ArrayIndex *)a;
        int mid = (pa->low + pa->high)/2;

        ArrayIndex aIndex[N];
        pthread_t thread[N];

        aIndex[0].low = pa->low;
        aIndex[0].high = mid;

        aIndex[1].low = mid+1;
        aIndex[1].high = pa->high;

        if (pa->low >= pa->high) return 0;

        int i;
        for(i = 0; i < N; i++) pthread_create(&thread[i], NULL, mergesort, &aIndex[i]);
        for(i = 0; i < N; i++) pthread_join(thread[i], NULL);

        merge(pa->low, pa->high);

        //pthread_exit(NULL);
        return 0;
}



int main()
{
  cout << "Init Array: " << endl;
  Print(a);
  //  for (int i = 0; i < 10; i++) printf ("%d ", a[i]);

  ArrayIndex ai;
  ai.low = 0;
  //ai.high = sizeof(a)/sizeof(a[0])-1;
  ai.high = a.size() - 1;
  pthread_t thread;

  pthread_create(&thread, NULL, mergesort, &ai);
  pthread_join(thread, NULL);

  cout << "\nSorted Array: " << endl;
  //for (int i = 0; i < 10; i++) printf ("%d ", a[i]);

  Print(a);
  cout << endl;

  return 0;
}
