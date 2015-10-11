#include <thrust/device_vector.h>

#define MAX_LABELS 256

void cudaJFAApolloniusThrust(cudaArray_t imageArray, int imageSize,
                             int numLabels,
                             thrust::device_vector<float4> &seedbuffer,
                             thrust::device_vector<float> &distance_vector,
                             thrust::device_vector<int> &compute_vector,
                             thrust::device_vector<int> &compute_temp_vector,
                             thrust::device_vector<int> &compute_seed_ids,
                             thrust::device_vector<int> &compute_seed_indices);

void cudaJFADistanceTransformThrust(
    cudaGraphicsResource_t &inputImage, cudaGraphicsResource_t &outputImage,
    int image_size, int screen_size_x, int screen_size_y,
    thrust::device_vector<int> &compute_vector,
    thrust::device_vector<float> &result_vector);

void cudaJFADistanceTransformThrust(
    cudaArray_t &inputImageArray, cudaChannelFormatDesc inputImageDesc,
    cudaArray_t &outputImageArray, int image_size, int screen_size_x,
    int screen_size_y, thrust::device_vector<int> &compute_vector,
    thrust::device_vector<float> &result_vector);
