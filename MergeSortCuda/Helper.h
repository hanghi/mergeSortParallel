#pragma once
#include <time.h>
#include <string>
#include <iostream>
#include <chrono>
#include <limits>
#include <fstream>
using namespace std;
using namespace std::chrono;
//number precision
typedef std::numeric_limits< double > dbl;
//Helper functions
////Generate sample
bool generateRandomList(int, long*&);
////write log to file
void writeLog(string, int, double, double, double);
////Show array to screen
void showArray(long*, int);
