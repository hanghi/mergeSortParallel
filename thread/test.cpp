#include <stdio.h>
#include <vector>
#include <stdlib.h>
#include <pthread.h>
#include <iostream>
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

void *convert() {
  vector<int> A = {1,2,3};
  return static_cast<void*>(&A);
}

int main() {
  vector<int> lines = {3,5,7,0,1,2,3,4,6,7,8,9,1,2,3,4,5,6,7,8};
  size_t const half_size = lines.size() / 2;

  int step = lines.size() / 4;
  int begin = 0;
  int end;

  cout << "lines" << step << endl;
  Print(lines);

  for(int i = 0; i < 4; i++) {
    vector<int> split_lo(lines.begin() + begin, lines.begin() + begin + step);
    begin += step;
    cout << "\nsplit_lo " << i << endl;
    Print(split_lo);
  }
    

  void *result = convert();
  vector<int> *BB= static_cast<vector<int>*>(result);
  Print(*BB);
  
  return 0;
}
