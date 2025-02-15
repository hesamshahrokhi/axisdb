#include "../include/lib.hpp"

void Timer::reset(){
    start = std::chrono::high_resolution_clock::now();
};

void Timer::print(std::string title){
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "======== Time [" << title << "] -> " << duration.count() << " ms" << std::endl;
};

template <typename T>
size_t Collection<T>::id_counter = 1;
void euclidean_norm(Vector<float>& vector, float& result) {
    auto size = vector.size;
    result = 0.0f;

    // Parallel computation using TBB
    tbb::enumerable_thread_specific<float> partial_result;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, size), [&](tbb::blocked_range<size_t> r) {
        float tmp = 0.0f;
        for (size_t j = r.begin(); j < r.end(); j++) {
            tmp += std::pow(vector.data[j], 2);
        }
        partial_result.local() += tmp;
    });
    result = sqrt(partial_result.combine(std::plus<float>()));
}

void dot_product(Vector<float>& v1, Vector<float>& v2, float& result) {
    auto size = v1.size;
    // Parallel computation using TBB
    tbb::enumerable_thread_specific<float> partial_result;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, size), [&](tbb::blocked_range<size_t> r) {
        float tmp = 0.0f;
        for (size_t j = r.begin(); j < r.end(); j++) {
            tmp += v1.data[j] * v2.data[j];
        }
        partial_result.local() = tmp;
    });
    result = partial_result.combine(std::plus<float>());
}

void euclidean_distance(Vector<float>& v1, Vector<float>& v2, float& result) {
    size_t size = v1.size;


    //// TBB + AVX2
    // result = tbb::parallel_reduce(
    //     tbb::blocked_range<size_t>(0, size), // Use a reasonable chunk size
    //     0.0f,
    //     [&](const tbb::blocked_range<size_t>& r, float local_sum) -> float {
    //         size_t i = r.begin();
    //         __m256 sum = _mm256_setzero_ps(); // AVX2 accumulator

    //         // Vectorized loop
    //         for (; i + 8 <= r.end(); i += 8) {
    //             __m256 a = _mm256_loadu_ps(&v1.data[i]);
    //             __m256 b = _mm256_loadu_ps(&v2.data[i]);

    //             __m256 diff = _mm256_sub_ps(a, b);
    //             __m256 square = _mm256_mul_ps(diff, diff);

    //             sum = _mm256_add_ps(sum, square);
    //         }

    //         // Store AVX2 result into scalar temp array
    //         float tmp[8];
    //         _mm256_storeu_ps(tmp, sum);
    //         local_sum += tmp[0] + tmp[1] + tmp[2] + tmp[3] +
    //                      tmp[4] + tmp[5] + tmp[6] + tmp[7];

    //         // Process remaining elements
    //         for (; i < r.end(); ++i) {
    //             float diff = v1.data[i] - v2.data[i];
    //             local_sum += diff * diff;
    //         }

    //         return local_sum;
    //     },
    //     std::plus<float>() // Reduction operation
    // );
    // result = std::sqrt(result);

    //// AVX2
    float final = 0.0f;
    // AVX2 vectorization
    size_t i = 0;
    __m256 sum = _mm256_setzero_ps();
    __m256 tmp = _mm256_setzero_ps();
    while (i + 8 <= size) {
        // Read data from memory
        __m256 a = _mm256_loadu_ps(&v1.data[i]);
        __m256 b = _mm256_loadu_ps(&v2.data[i]);
        // Compute difference
        tmp = _mm256_sub_ps(a, b);
        // Compute square
        tmp = _mm256_mul_ps(tmp, tmp);
        // Accumulate sum
        sum = _mm256_add_ps(tmp, sum);
        i += 8;
    }
    // Add 8 32-bit elements of sum
    float tmp2[8];
    _mm256_storeu_ps(tmp2, sum);
    final += tmp2[0] + tmp2[1] + tmp2[2] + tmp2[3] + tmp2[4] + tmp2[5] + tmp2[6] + tmp2[7];
    // Add remaining elements of sum
    while (i < size) {
        final += std::pow(v1.data[i] - v2.data[i], 2);
        i++;
    }
    result = sqrt(final);



    //// Sequential
    // float tmp = 0.0f;
    // for (size_t j = 0; j < size; j++) {
    //     tmp += std::pow(v1.data[j] - v2.data[j], 2);
    // }
    // result = sqrt(tmp);

}

void cosine_similarity(Vector<float>& v1, Vector<float>& v2, float& result) {
    dot_product(v1, v2, result);
    result = result / (v1.norm * v2.norm);
}

template <typename T>
void Collection<T>::populate(std::unique_ptr<T[]> _data, size_t vectors_count, size_t vector_size) {
    this->size = vectors_count;
    this->vector_size = vector_size;
    this->vectors = std::make_unique<Vector<T>[]>(vectors_count);
    for (size_t i = 0; i < vectors_count; i++) {
        this->vectors[i].id = id_counter++;
        this->vectors[i].size = vector_size;
        this->vectors[i].data = std::make_unique<T[]>(vector_size);
        for (size_t j = 0; j < vector_size; j++) {
            this->vectors[i].data[j] = _data[i * vector_size + j];
        }
        euclidean_norm(this->vectors[i], this->vectors[i].norm);
    }
};

template <typename T>
void print_vector(Vector<T>& vector){
    auto size = vector.size;
    std::cout << "[";
    for (size_t j = 0; j < size; j++) {
        std::cout << vector.data[j];
        if (j < size - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}
template void print_vector<float>(Vector<float>&);

void read_hdf5_dataset(const char* filename, const char* dataset_name, Collection<float>& collection)
{
    using namespace H5;

    try {
        H5File file(filename, H5F_ACC_RDONLY);
        DataSet dataset = file.openDataSet(dataset_name);
        DataSpace dataspace = dataset.getSpace();
        hsize_t dims[2];
        dataspace.getSimpleExtentDims(dims, NULL);
        size_t num_vectors = dims[0];
        size_t vector_size = dims[1];
        auto tmp_data = std::make_unique<float[]>(num_vectors * vector_size);
        dataset.read(tmp_data.get(), PredType::NATIVE_FLOAT);
        collection.populate(std::move(tmp_data), num_vectors, vector_size);
    } catch (Exception& e) {
        std::cerr << "### HDF5 error: " << e.getDetailMsg() << std::endl;
    }
}