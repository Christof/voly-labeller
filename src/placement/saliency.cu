#if _WIN32
#pragma warning(disable : 4244)
#endif

#include "./saliency.h"
#include "../utils/cuda_helper.h"

inline __host__ __device__ float3 operator+(float3 a, float3 b)
{
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ float3 operator-(float3 a, float3 b)
{
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ float3 operator-(float3 a)
{
  return make_float3(-a.x, -a.y, -a.z);
}

inline __host__ __device__ float operator*(float3 a, float3 b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ float3 operator*(float a, float3 b)
{
  return make_float3(a * b.x, a * b.y, a * b.z);
}

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
  float yr = 1.0;       // reference white
  float zr = 1.088754;  // reference white

  float fx = f(xyz.x / xr);
  float fy = f(xyz.y / yr);
  float fz = f(xyz.z / zr);

  return make_float3(116.0f * fy - 16.0f, 500.0f * (fx - fy),
                     200.0f * (fy - fz));
}

__device__ float3 rgbaToLab(float4 rgba)
{
  return xyzToLab(rgbToXyz(make_float3(rgba.x, rgba.y, rgba.z)));
}

__device__ float rgbaToLightness(float4 rgba)
{
  float3 xyz = rgbToXyz(make_float3(rgba.x, rgba.y, rgba.z));
  float fy = f(xyz.y);
  return 116.0f * fy - 16.0f;
}

__global__ void saliencyKernel(cudaTextureObject_t input,
                               cudaSurfaceObject_t output, int width,
                               int height, int widthScale, int heightScale)
{
  int xOutput = blockIdx.x * blockDim.x + threadIdx.x;
  int yOutput = blockIdx.y * blockDim.y + threadIdx.y;
  if (xOutput >= width || yOutput >= height)
    return;

  float x = xOutput * widthScale;
  float y = yOutput * heightScale;

  float leftUpper =
    rgbaToLightness(tex2D<float4>(input, x - 1 + 0.5f, y - 1 + 0.5f));
  float left = rgbaToLightness(tex2D<float4>(input, x - 1 + 0.5f, y + 0.5f));
  float leftLower =
    rgbaToLightness(tex2D<float4>(input, x - 1 + 0.5f, y + 1 + 0.5f));

  float upper = rgbaToLightness(tex2D<float4>(input, x + 0.5f, y - 1 + 0.5f));
  float lower = rgbaToLightness(tex2D<float4>(input, x + 0.5f, y + 1 + 0.5f));

  float rightUpper =
    rgbaToLightness(tex2D<float4>(input, x + 1 + 0.5f, y - 1 + 0.5f));
  float right = rgbaToLightness(tex2D<float4>(input, x + 1 + 0.5f, y + 0.5f));
  float rightLower =
    rgbaToLightness(tex2D<float4>(input, x + 1 + 0.5f, y + 1 + 0.5f));

  float resultX = -leftUpper - 2.0f * left - leftLower + rightUpper +
                   2.0f * right + rightLower;
  float resultY = -leftUpper - 2.0f * upper - rightUpper + leftLower +
                   2.0f * lower + rightLower;

  float magnitudeSquared = resultX * resultX + resultY * resultY;
  surf2Dwrite(1e-5f * magnitudeSquared, output, xOutput * sizeof(float),
              yOutput);
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

  saliencyKernel<<<dimGrid, dimBlock>>>(input, output,
      outputWidth, outputHeight, widthScale, heightScale);

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
