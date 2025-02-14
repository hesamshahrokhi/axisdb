#include <H5Cpp.h>
#include <iostream>
#include <vector>

using namespace H5;
using namespace std;

const std::string FILE_NAME = "/home/ubuntu/data/fashion/fashion-mnist-784-euclidean.hdf5";
const std::string DATASET_NAME = "/train";  // Path to dataset

int main() {
    try {
        // Open the HDF5 file
        H5File file(FILE_NAME, H5F_ACC_RDONLY);

        // Open the dataset
        DataSet dataset = file.openDataSet(DATASET_NAME);

        // Get dataset dimensions
        DataSpace dataspace = dataset.getSpace();
        hsize_t dims[2];  // For a 2D array (60000, 784)
        dataspace.getSimpleExtentDims(dims, NULL);

        size_t num_vectors = dims[0]; // Number of vectors (60,000)
        size_t vector_size = dims[1]; // Size of each vector (784)

        cout << "Dataset size: " << num_vectors << " vectors, each of size " << vector_size << endl;

        // Prepare memory for reading the first vector
        std::vector<float> first_vector(vector_size); 

        // Define hyperslab to read only the first row
        hsize_t offset[2] = {0, 0};   // Start at first vector
        hsize_t count[2] = {1, vector_size};  // Read 1 vector of size 784
        dataspace.selectHyperslab(H5S_SELECT_SET, count, offset);

        // Define memory space to match hyperslab
        DataSpace memspace(2, count);

        // Read data into vector
        dataset.read(first_vector.data(), PredType::NATIVE_FLOAT, memspace, dataspace);

        // Print first 10 values of the first vector
        cout << "First 10 values of the first vector:\n";
        for (size_t i = 0; i < vector_size; i++) {
            cout << first_vector[i] << " ";
        }
        cout << "..." << endl;

    } catch (const Exception &error) {
        cerr << "HDF5 Error: " << error.getDetailMsg() << endl;
        return 1;
    }

    return 0;
}
