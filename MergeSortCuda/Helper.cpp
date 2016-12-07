#include "Helper.h"

//Generate unsorted list
bool generateRandomList(long length, int*& list) {

	list = (int*)malloc(sizeof(int) * length);
	if (list == NULL)
		return false;

	for (int i = 0; i < length; i++)	
		list[i] = rand() % length ;
	
	return true;
}

//save result test to file
void writeLog(string fileLog, int length, double time1, double time2, double time3, double time4, double time5) {
	std::ofstream log(fileLog, std::ios_base::app | std::ios_base::out);
	log.precision(dbl::max_digits10);
	log << length << "\t" << fixed << time1 << "\t" << fixed << time2 << "\t" << fixed << time3 << "\t" << fixed << time4 << "\t" << fixed << time5 << endl;
}

//print array to screen
void showArray(int *a, long l) {
	for (int i = 0; i < l; i++) {
		cout << a[i] << "  ";
	}
	cout << endl;
}