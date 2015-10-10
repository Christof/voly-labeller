#include "./distance_transform.h"
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include "../utils/cuda_helper.h"

surface<void, cudaSurfaceType2D> surfaceWrite;
texture<float, 2, cudaReadModeElementType> depthTexture;

__global__ void jfa_seed_kernel(int imageSize, int num_labels,
                                float4 *seedbuffer, int *thrustptr, int *idptr,
                                int *idxptr)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= imageSize || y >= imageSize)
    return;

  int index = y * imageSize + x;
  float4 outval = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
  float4 seedval = make_float4(0.0f, 0.0f, 0.0f, 1.0f);

  // initialize to out of bounds
  int outindex = (imageSize * 2) * (imageSize * 2) - 1;

  for (int i = 0; i < num_labels; i++)
  {
    float4 seedval = seedbuffer[i];
    // if (seedval.x > 0.0) outval = make_float4(0.0f, 1.0f, 0.0f, 1.0f);
    if (int(seedval.x) > 0 && x == int(seedval.y) && y == int(seedval.z) &&
        (x != 0 || y != 0))
    {
      outval = make_float4(seedval.x / (num_labels + 1),
                           int(seedval.y) / float(imageSize),
                           int(seedval.z) / float(imageSize), 1.0f);

      // index for thrust computation =
      outindex = x + y * imageSize;
      // printf("hit: outval: %f %f %f %f %d %d %f %f %f nl: %d oi:
      // %d\n",outval.x, outval.y, outval.z, outval.w, x, y, seedval.x,
      // seedval.y, seedval.z,num_labels, outindex);
      // printf("hit: outval: %f %f %f %f %d %d\n",outval.x, outval.y, outval.z,
      // outval.w, x, y);
      // break;
    }
    else
    {
      // printf("not hit: outval: %f %f %f %f %d %d %f %f %f \n",outval.x,
      // outval.y, outval.z, outval.w, x, y, seedval.x, seedval.y, seedval.z);
    }
    idptr[i] = int(seedval.x);
    idxptr[i] = int(seedval.y) + int(seedval.z) * imageSize;
  }

  thrustptr[index] = outindex;
  surf2Dwrite<float4>(outval, surfaceWrite, x * sizeof(float4), y);
}

__global__ void apollonius_cuda(int *data, float *occupancy, unsigned int step,
                                int w, int h)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h)
    return;

  int index = y * w + x;

  int cur_nearest = data[index];
  int cur_y = cur_nearest / w;
  int cur_x = cur_nearest - cur_y * w;
  float curr_w = (cur_nearest < w * h) ? occupancy[cur_nearest] : 0.0f;

  float cur_dist =
      sqrtf(float((x - cur_x) * (x - cur_x) + (y - cur_y) * (y - cur_y))) -
      curr_w;

#pragma unroll
  for (int i = -1; i <= 1; i++)
  {
    int u = x + i * step;
    if (u < 0 || u >= w)
      continue;
#pragma unroll
    for (int j = -1; j <= 1; j += 2 - i * i)
    {
      int v = y + j * step;
      if (v < 0 || v >= h)
        continue;

      int newindex = v * w + u;
      int new_nearest = data[newindex];
      int new_y = new_nearest / w;
      int new_x = new_nearest - new_y * w;
      float new_w = (new_nearest < w * h) ? occupancy[new_nearest] : 0.0f;
      float new_dist =
          sqrtf(float((x - new_x) * (x - new_x) + (y - new_y) * (y - new_y))) -
          new_w;

      if (new_dist < cur_dist || cur_nearest >= w * h)
      {
        cur_dist = new_dist;
        cur_nearest = new_nearest;
      }
    }
  }

  data[index] = cur_nearest;
}

__global__ void /*__launch_bounds__(16)*/ jfa_thrust_gather(int imageSize,
                                                            int num_labels,
                                                            int *thrustptr,
                                                            int *seedidptr,
                                                            int *seedidxptr)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= imageSize || y >= imageSize)
    return;
  int index = y * imageSize + x;
  float4 color;
  int labelID = 100;
  int labelIndex = thrustptr[index];
  // int ly = labelID/imageSize;
  // int lx = labelID - ly*imageSize;

  // bool foundcolor = false;
  for (int i = 0; i < num_labels; i++)
  {
    // int sly = seedptr[i]/imageSize;
    // int slx = seedptr[i] - sly*imageSize;
    // if (slx == lx && sly == ly)
    //{
    if (labelIndex == seedidxptr[i])
    {
      // foundcolor = true;
      labelID = seedidptr[i];
      break;
    }
  }

  switch (labelID)
  {
  case 0:
    color = make_float4(0.0, 0.0, 0.0, 1.0);
    break;
  case 1:
    color = make_float4(1.0, 0.0, 0.0, 1.0);
    break;
  case 2:
    color = make_float4(0.0, 1.0, 0.0, 1.0);
    break;
  case 3:
    color = make_float4(0.0, 0.0, 1.0, 1.0);
    break;
  case 4:
    color = make_float4(1.0, 1.0, 0.0, 1.0);
    break;
  case 5:
    color = make_float4(0.0, 1.0, 1.0, 1.0);
    break;
  case 6:
    color = make_float4(1.0, 0.0, 1.0, 1.0);
    break;
  case 7:
    color = make_float4(1.0, 1.0, 1.0, 1.0);
    break;
  default:
    color = make_float4(0.5, 0.5, 0.5, 1.0);
  }
  surf2Dwrite<float4>(color, surfaceWrite, x * sizeof(float4), y);
}

/**
 * \brief Initializes the distance transform
 *
 * The value from the depthTexture is read. If it is larger than or equel to
 * 0.99 the data value is set to the index. Otherwise it is set to the given
 * outlier value.
 */
__global__ void initializeForDistanceTransform(int width, int height,
    float xscale, float yscale, int outlierValue, int *data)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  int index = y * width + x;

  float pixelValue = tex2D(depthTexture, x * xscale + 0.5f, y * yscale + 0.5f);

  data[index] = pixelValue >= 0.99f ? index : outlierValue;
}

__global__ void jfa_cuda(int *data, unsigned int step, int w, int h)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h)
    return;

  int index = y * w + x;

  int cur_nearest = data[index];
  int cur_y = cur_nearest / w;
  int cur_x = cur_nearest - cur_y * w;
  int cur_dist = (x - cur_x) * (x - cur_x) + (y - cur_y) * (y - cur_y);

// if (x>253 && x<257 && y==255)
//  if ((x==255 && y==255) || (x==10 && y==10))
//  {
//    printf("%d %d: step: %d cur: %d %d %d, %d %d\n", x, y, step, cur_nearest,
//    cur_x, cur_y, w, h);
//  }

#pragma unroll
  for (int i = -1; i <= 1; i++)
  {
    int u = x + i * step;
    if (u < 0 || u >= w)
      continue;
#pragma unroll
    for (int j = -1; j <= 1; j += 2 - i * i)
    {
      int v = y + j * step;
      if (v < 0 || v >= h)
        continue;

      int newindex = v * w + u;
      int new_nearest = data[newindex];
      int new_y = new_nearest / w;
      int new_x = new_nearest - new_y * w;
      int new_dist = (x - new_x) * (x - new_x) + (y - new_y) * (y - new_y);

      if (new_dist < cur_dist)
      {
        cur_dist = new_dist;
        cur_nearest = new_nearest;
      }
    }
  }

  data[index] = cur_nearest;
}

__global__ void jfa_dtf_compute_distance_kernel(int image_size, int *index_ptr,
                                                float *value_ptr)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= image_size || y >= image_size)
    return;

  int index = y * image_size + x;
  int voronoival = index_ptr[index];

  int ty = voronoival / image_size;
  int tx = voronoival - ty * image_size;

  float sqdist = ((tx - x) * (tx - x) + (ty - y) * (ty - y));
  // float dist = sqdist;
  float distf = sqrtf(sqdist);

  value_ptr[index] = distf;

  // write to texture for debugging
  float4 color =
      make_float4(16.0f * distf / image_size, 16.0f * distf / image_size,
                  16.0f * distf / image_size, 1.0f);
  // float4 color = make_float4(512.0f*distf/image_size,
  // 256.0f*distf/image_size, 16.0f*distf/image_size, 1.0f);
  surf2Dwrite<float4>(color, surfaceWrite, x * sizeof(float4), y);
}

void cudaJFADistanceTransformThrust(
    cudaGraphicsResource_t &inputImage, cudaGraphicsResource_t &outputImage,
    int image_size, int screen_size_x, int screen_size_y,
    thrust::device_vector<int> &compute_vector,
    thrust::device_vector<int> &compute_temp_vector,
    thrust::device_vector<float> &result_vector)
{
  cudaGraphicsMapResources(1, &inputImage);
  cudaArray_t inputImageArray;
  cudaGraphicsSubResourceGetMappedArray(&inputImageArray, inputImage, 0, 0);
  cudaChannelFormatDesc inputImageDesc;
  cudaGetChannelDesc(&inputImageDesc, inputImageArray);

  cudaGraphicsMapResources(1, &outputImage);
  cudaArray_t outputImageArray;
  cudaGraphicsSubResourceGetMappedArray(&outputImageArray, outputImage, 0, 0);

  cudaJFADistanceTransformThrust(inputImageArray, inputImageDesc,
                                 outputImageArray, image_size, screen_size_x,
                                 screen_size_y, compute_vector,
                                 compute_temp_vector, result_vector);

  cudaGraphicsUnmapResources(1, &outputImage);
  cudaGraphicsUnmapResources(1, &inputImage);
}

void cudaJFADistanceTransformThrust(
    cudaArray_t &inputImageArray, cudaChannelFormatDesc inputImageDesc,
    cudaArray_t &outputImageArray, int image_size, int screen_size_x,
    int screen_size_y, thrust::device_vector<int> &compute_vector,
    thrust::device_vector<int> &compute_temp_vector,
    thrust::device_vector<float> &result_vector)
{
  if (compute_vector.size() !=
      static_cast<unsigned long>(image_size * image_size))
  {
    compute_vector.resize(image_size * image_size, image_size * image_size);
    compute_temp_vector.resize(image_size * image_size,
                               image_size * image_size);
    result_vector.resize(image_size * image_size, image_size * image_size);
  }

  int *compute_index_ptr = thrust::raw_pointer_cast(compute_vector.data());
  float *result_value_ptr = thrust::raw_pointer_cast(result_vector.data());

  dim3 dimBlock(32, 1, 1);
  dim3 dimGrid(divUp(image_size, dimBlock.x), divUp(image_size, dimBlock.y), 1);
  {
    // read depth buffer and initialize distance transform computation
    depthTexture.normalized = 0;
    depthTexture.filterMode = cudaFilterModeLinear /*cudaFilterModePoint*/;
    depthTexture.addressMode[0] = cudaAddressModeWrap;
    depthTexture.addressMode[1] = cudaAddressModeWrap;

    cudaBindTextureToArray(&depthTexture, inputImageArray, &inputImageDesc);
    float xScale = static_cast<float>(screen_size_x) / image_size;
    float yScale = static_cast<float>(screen_size_y) / image_size;
    int outlierValue = (image_size * 2) * (image_size * 2) - 1;

    initializeForDistanceTransform<<<dimGrid, dimBlock>>>(image_size,
        image_size, xScale, yScale, outlierValue, compute_index_ptr);
    cudaThreadSynchronize();

    cudaUnbindTexture(&depthTexture);
  }

  // voronoi diagram computation in thrust
  {
    // compute_temp_vector = compute_vector;
    jfa_cuda << <dimGrid, dimBlock>>>
        (thrust::raw_pointer_cast(compute_vector.data()), 1, image_size,
         image_size);
    // compute_vector.swap(compute_temp_vector);

    // cudaThreadSynchronize();
    for (int k = (image_size / 2); k > 0; k /= 2)
    {
      jfa_cuda<<<dimGrid, dimBlock>>>(
          thrust::raw_pointer_cast(compute_vector.data()), k, image_size,
          image_size);
      // cudaThreadSynchronize();
      // compute_vector.swap(compute_temp_vector);
    }

    cudaThreadSynchronize();
  }

  {
    cudaChannelFormatDesc outputChannelDesc =
        cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    HANDLE_ERROR(cudaBindSurfaceToArray(surfaceWrite, outputImageArray,
                                        outputChannelDesc));

    // kernel which maps color to distance transform result
    compute_index_ptr = thrust::raw_pointer_cast(compute_vector.data());
    jfa_dtf_compute_distance_kernel<<<dimGrid, dimBlock>>>(
        image_size, compute_index_ptr, result_value_ptr);

    cudaThreadSynchronize();
  }
}

void cudaJFAApolloniusThrust(cudaArray_t imageArray, int imageSize,
                             int numLabels,
                             thrust::device_vector<float4> &seedbuffer,
                             thrust::device_vector<float> &distance_vector,
                             thrust::device_vector<int> &compute_vector,
                             thrust::device_vector<int> &compute_temp_vector,
                             thrust::device_vector<int> &compute_seed_ids,
                             thrust::device_vector<int> &compute_seed_indices)
{
  int pixelCount = imageSize * imageSize;
  if (compute_vector.size() != static_cast<unsigned long>(pixelCount))
  {
    compute_vector.resize(pixelCount, pixelCount);
    compute_temp_vector.resize(pixelCount, pixelCount);
  }

  if (compute_seed_ids.size() != MAX_LABELS ||
      compute_seed_indices.size() != MAX_LABELS)
  {
    compute_seed_ids.resize(MAX_LABELS, -1);
    compute_seed_indices.resize(MAX_LABELS, -1);
  }

  int *raw_ptr = thrust::raw_pointer_cast(compute_vector.data());
  int *idptr = thrust::raw_pointer_cast(compute_seed_ids.data());
  int *idxptr = thrust::raw_pointer_cast(compute_seed_indices.data());
  float4 *seedBufferPtr = thrust::raw_pointer_cast(seedbuffer.data());

  dim3 dimBlock(32, 32, 1);
  dim3 dimGrid(divUp(imageSize, dimBlock.x), divUp(imageSize, dimBlock.y), 1);

  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
  HANDLE_ERROR(cudaBindSurfaceToArray(surfaceWrite, imageArray, channelDesc));

  jfa_seed_kernel << <dimGrid, dimBlock>>>
      (imageSize, numLabels, seedBufferPtr, raw_ptr, idptr, idxptr);
  HANDLE_ERROR(cudaThreadSynchronize());

  compute_temp_vector = compute_vector;
  apollonius_cuda << <dimGrid, dimBlock>>>
      (thrust::raw_pointer_cast(compute_vector.data()),
       thrust::raw_pointer_cast(distance_vector.data()), 1, imageSize,
       imageSize);
  /*

  for (int k = (imageSize / 2); k > 0; k /= 2)
  {
    apollonius_cuda<<<dimGrid, dimBlock>>>(
        thrust::raw_pointer_cast(compute_vector.data()),
        thrust::raw_pointer_cast(distance_vector.data()), k, imageSize,
        imageSize);
  }

  // colorize diagram
  jfa_thrust_gather<<<dimGrid, dimBlock>>>(imageSize, numLabels, raw_ptr,
      idptr, idxptr);
  cudaThreadSynchronize();
  */
}

