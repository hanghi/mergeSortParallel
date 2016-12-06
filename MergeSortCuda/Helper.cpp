#include "Helper.h"

//Generate unsorted list
bool generateRandomList(int length, long*& list) {

	list = (long*)malloc(sizeof(long) * length);
	if (list == NULL)
		return false;

	for (int i = 0; i < length; i++)	
		list[i] = rand() % length;
	
	return true;
}

//save result test to file
void writeLog(string fileLog, int length, double time1, double time2, double time3) {
	std::ofstream log(fileLog, std::ios_base::app | std::ios_base::out);
	log.precision(dbl::max_digits10);
	log << length << "\t" << fixed << time1 << "\t" << fixed << time2 << "\t" << fixed << time3 << endl;
}

//print array to screen
void showArray(long *a, int l) {
	for (int i = 0; i < l; i++) {
		cout << a[i] << "  ";
	}
	cout << endl;
}
