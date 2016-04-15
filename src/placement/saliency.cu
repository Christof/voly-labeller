#include "./saliency.h"
#include "../utils/cuda_helper.h"

__device__ float3 rgbToXyz(float3 rgb)
{
  float r, g, b;
  if (rgb.x <= 0.04045f)
    r = rgb.x / 12.92f;
  else
    r = pow((rgb.x + 0.055f) / 1.055f, 2.4f);
  if (rgb.y <= 0.04045f)
    g = rgb.y / 12.92f;
  else
    g = pow((rgb.y + 0.055f) / 1.055f, 2.4f);
  if (rgb.z <= 0.04045f)
    b = rgb.z / 12.92f;
  else
    b = pow((rgb.z + 0.055f) / 1.055f, 2.4f);

  float x = r * 0.4124564f + g * 0.3575761f + b * 0.1804375f;
  float y = r * 0.2126729f + g * 0.7151522f + b * 0.0721750f;
  float z = r * 0.0193339f + g * 0.1191920f + b * 0.9503041f;

  return make_float3(x, y, z);
}

__device__ float f(float t)
{
  const float epsilon = 0.008856452f;
  const float kappa = (29.0f / 6.0f) * (29.0f / 6.0f) / 3.0f;

  if (t > epsilon)
    return pow(t, 1.0f / 3.0f);

  return kappa * t + 4.0f / 29.0f;
}

__device__ float3 xyzToLab(float3 xyz)
{
  float xr = 0.950456;  // reference white
  float yr = 1.0;  // reference white
  float zr = 1.088754;  // reference white


  float fx = f(xyz.x / xr);
  float fy = f(xyz.y / yr);
  float fz = f(xyz.z / zr);

  return make_float3(116.0f * fy - 16.0f, 500.0f * (fx - fy),
                     200.0f * (fy - fz));
}

__global__ void saliencyKernel(cudaTextureObject_t input,
                               cudaSurfaceObject_t output, int width,
                               int height, int widthScale, int heightScale)
{
  int xOutput = blockIdx.x * blockDim.x + threadIdx.x;
  int yOutput = blockIdx.y * blockDim.y + threadIdx.y;
  if (xOutput >= width || yOutput >= height)
    return;

  // TODO (SIR)
  // - downsampling
  // - handle colors
  float x = xOutput * widthScale;
  float y = yOutput * heightScale;

  float4 leftUpper = tex2D<float4>(input, x - 1 + 0.5f, y - 1 + 0.5f);
  float4 left = tex2D<float4>(input, x - 1 + 0.5f, y + 0.5f);
  float4 leftLower = tex2D<float4>(input, x - 1 + 0.5f, y + 1 + 0.5f);

  float4 upper = tex2D<float4>(input, x + 0.5f, y - 1 + 0.5f);
  float4 lower = tex2D<float4>(input, x + 0.5f, y + 1 + 0.5f);

  float4 rightUpper = tex2D<float4>(input, x + 1 + 0.5f, y - 1 + 0.5f);
  float4 right = tex2D<float4>(input, x + 1 + 0.5f, y + 0.5f);
  float4 rightLower = tex2D<float4>(input, x + 1 + 0.5f, y + 1 + 0.5f);

  float resultX = -leftUpper.x - 2.0f * left.x - leftLower.x +
                  rightUpper.x + 2.0f * right.x + rightLower.x;
  float resultY = -leftUpper.x - 2.0f * upper.x - rightUpper.x +
                 leftLower.x + 2.0f * lower.x + rightLower.x;

  float magnitudeSquared = resultX * resultX + resultY * resultY;
  surf2Dwrite(magnitudeSquared, output, xOutput * sizeof(float), yOutput);
}

namespace Placement
{

Saliency::Saliency(std::shared_ptr<CudaArrayProvider> inputProvider,
                   std::shared_ptr<CudaArrayProvider> outputProvider)
  : inputProvider(inputProvider), outputProvider(outputProvider)
{
}

Saliency::~Saliency()
{
  if (input)
    cudaDestroyTextureObject(input);
  if (output)
    cudaDestroySurfaceObject(output);
}

void Saliency::runKernel()
{
  if (!input)
    createSurfaceObjects();

  float outputWidth = outputProvider->getWidth();
  float outputHeight = outputProvider->getHeight();

  dim3 dimBlock(32, 32, 1);
  dim3 dimGrid(divUp(outputWidth, dimBlock.x), divUp(outputHeight, dimBlock.y),
               1);

  int widthScale = inputProvider->getWidth() / outputWidth;
  int heightScale = inputProvider->getHeight() / outputHeight;

  saliencyKernel<<<dimGrid, dimBlock>>>
      (input, output, outputWidth, outputHeight, widthScale, heightScale);

  HANDLE_ERROR(cudaThreadSynchronize());
}

void Saliency::createSurfaceObjects()
{
  inputProvider->map();
  outputProvider->map();

  auto resDesc = inputProvider->getResourceDesc();
  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeBorder;
  texDesc.addressMode[1] = cudaAddressModeBorder;
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;

  cudaCreateTextureObject(&input, &resDesc, &texDesc, NULL);

  auto outputResDesc = outputProvider->getResourceDesc();
  cudaCreateSurfaceObject(&output, &outputResDesc);

  inputProvider->unmap();
  outputProvider->unmap();
}

}  // namespace Placement
