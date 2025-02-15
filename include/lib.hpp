#ifndef LIB_HPP
#define LIB_HPP


#include <iostream>
#include <cstddef>
#include <memory>
#include <cmath>
#include <chrono>

#include <H5Cpp.h>
#include <tbb/tbb.h>

//////////////////////// GLOBAL VARIABLES
// extern size_t PREFETCH_SIZE;

//////////////////////// Data Structures

struct Timer
{
    std::chrono::high_resolution_clock::time_point start;
    void reset();
    void print(std::string title);
};


template <typename T> 
struct Vector {
    size_t id;
    std::unique_ptr<T[]> data;
    size_t size;
    float norm;
};

template <typename T>
struct Collection {
    static size_t id_counter;

    std::string name;
    size_t size;
    size_t vector_size;
    std::unique_ptr<Vector<T>[]> vectors;
    
    Collection(std::string name) : name(name) {}
    void populate(std::unique_ptr<T[]> _data, size_t vectors_count, size_t vector_size);
};


//////////////////////// Helper Functions

template <typename T>
void print_vector(Vector<T>& vector);

//////////////////////// Data Preparation Functions

// Read HDF5 dataset into memory
void read_hdf5_dataset(const char* filename, const char* dataset_name, Collection<float>& collection);

//////////////////////// Vector Similarity Functions

void euclidean_norm(Vector<float>& vector, float& result);

void dot_product(Vector<float>& v1, Vector<float>& v2, float& result);

void euclidean_distance(Vector<float>& v1, Vector<float>& v2, float& result);

void cosine_similarity(Vector<float>& v1, Vector<float>& v2, float& result);

#endif // LIB_H