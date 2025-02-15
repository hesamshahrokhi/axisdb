#include "../include/lib.hpp"

const std::string FILE_NAME = "/home/ubuntu/data/fashion/fashion-mnist-784-euclidean.hdf5";
const std::string DATASET_NAME = "/train";

int main() {

    Timer t;
    Collection<float> collection ("fashion-mnist-train");

    t.reset();
    read_hdf5_dataset(FILE_NAME.c_str(), DATASET_NAME.c_str(), collection);
    t.print("reading HDF5 dataset");

    auto num_vectors = collection.size;
    auto vector_size = collection.vector_size;

    std::cout << "Number of vectors: " << num_vectors << std::endl;
    std::cout << "Vector size: " << vector_size << std::endl;

    t.reset();
    float d1 = 0.0f;
    for (size_t i = 0; i < num_vectors - 1; i++) {
        float tmp = 0.0f;
        euclidean_distance(collection.vectors[i], collection.vectors[i+1], tmp);
        d1 += tmp;
    }
    t.print("euclidean distance");

    t.reset();
    float d2 = 0.0f;
    for (size_t i = 0; i < num_vectors - 1; i++) {
        float tmp = 0.0f;
        dot_product(collection.vectors[i], collection.vectors[i+1], tmp);
        d2 += tmp;
    }
    t.print("dot product");

    t.reset();
    float d3 = 0.0f;
    for (size_t i = 0; i < num_vectors - 1; i++) {
        float tmp = 0.0f;
        cosine_similarity(collection.vectors[i], collection.vectors[i+1], tmp);
        d3 += tmp;
    }
    t.print("cosine similarity");

    std::cout << "Euclidean distance: " << d1 << std::endl;
    std::cout << "Dot product: " << d2 << std::endl;
    std::cout << "Cosine similarity: " << d3 << std::endl;

    return 0;

}
