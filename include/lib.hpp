#ifndef LIB_H
#define LIB_H

#include <cstddef>

// GLOBAL VARIABLES
size_t PREFETCH_SIZE = 64;

// Vector Similarity Functions

double dot_product(double* v1, double* v2, int size);

double euclidean_distance(double* v1, double* v2, int size);

double cosine_similarity(double* v1, double* v2, int size, double norm1, double norm2);


#endif // LIB_H