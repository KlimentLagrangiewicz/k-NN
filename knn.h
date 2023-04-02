#ifndef KNN_H
#define KNN_H

#include <stdlib.h>
#include <string.h>
#include <math.h>

void autoscaling(double *x, double *y, const int n1, const int n2, const int m);
double distEv(const double *x, const double *c, const int m);
int getNumOfClass(const int *y, const int n);
void InsertionSort(double *pr, int *cl, int count);
void MaxHeapify(double *pr, int *cl, int heapSize, int index);
void HeapSort(double *pr, int *cl, int count);
int Partition(double* pr, int *cl, int left, int right);
void QuickSortRecursive(double *pr, int *cl, int left, int right);
void IntroSort(double *pr, int *cl, int count);
void kNN(double *xtr, const int *y, double *xts, int *res, const int n, const int m, const int n2, const int k);

#endif
