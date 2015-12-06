#include "./summed_area_table.h"
#include <thrust/device_vector.h>
#include <iostream>
#include "../utils/cuda_helper.h"

texture<float, 2, cudaReadModeElementType> textureReadDepth;

#define WS 32    // Warp size (defines b x b block size where b = WS)
#define HWS 16   // Half Warp Size
#define DW 8     // Default number of warps (computational block height)
#define CHW 7    // Carry-heavy number of warps
                 // (computational block height for some kernels)
#define OW 6     // Optimized number of warps (computational block height for
                 // some kernels)
#define DNB 6    // Default number of blocks per SM (minimum blocks per SM
                 // launch bounds)
#define ONB 5    // Optimized number of blocks per SM (minimum blocks per SM
                 // for some kernels)
#define MTS 192  // Maximum number of threads per block with 8 blocks per SM
#define MBO 8    // Maximum number of blocks per SM using optimize or maximum
                 // warps
#define CHB 7    // Carry-heavy number of blocks per SM using default number of
                 // warps
#define MW 6     // Maximum number of warps per block with 8 blocks per SM (with
                 // all warps computing)
#define SOW 5    // Dual-scheduler optimized number of warps per block (with
                 // 8 blocks per SMand to use the dual scheduler with 1
                 // computing warp)
#define MBH 3    // Maximum number of blocks per SM using half-warp size

/**
 *  @ingroup gpu
 *  @brief Algorithm SAT stage 1
 *
 *  This function computes the algorithm stage S.1 following:
 *
 *  In parallel for all \f$m\f$ and \f$n\f$, compute and store the
 *  \f$P_{m,n}(\bar{Y})\f$ and \f$P^T_{m,n}(\hat{V})\f$.
 *
 *  @note The CUDA kernel functions (as this one) have many
 *  idiosyncrasies and should not be used lightly.
 *
 *  @see [Nehab:2011] cited in alg5() and figure in algSAT()
 *  @param[in] g_in Input image
 *  @param[out] g_ybar All \f$P_{m,n}(\bar{Y})\f$
 *  @param[out] g_vhat All \f$P^T_{m,n}(\hat{V})\f$
 */
__global__ void algSAT_stage1(const int c_width, const int c_height,
                              const float *g_in, float *g_ybar, float *g_vhat)
{
  const int tx = threadIdx.x, ty = threadIdx.y, bx = blockIdx.x,
            by = blockIdx.y, col = bx * WS + tx, row0 = by * WS;

  __shared__ float s_block[WS][WS + 1];

  float(*bdata)[WS + 1] = (float(*)[WS + 1]) & s_block[ty][tx];

  g_in += (row0 + ty) * c_width + col;
  g_ybar += by * c_width + col;
  g_vhat += bx * c_height + row0 + tx;

#pragma unroll
  for (int i = 0; i < WS - (WS % SOW); i += SOW)
  {
    **bdata = *g_in;
    bdata += SOW;
    g_in += SOW * c_width;
  }
  if (ty < WS % SOW)
  {
    **bdata = *g_in;
  }

  __syncthreads();

  if (ty == 0)
  {
    {  // calculate ybar -----------------------
      float(*bdata)[WS + 1] = (float(*)[WS + 1]) & s_block[0][tx];

      float prev = **bdata;
      ++bdata;

#pragma unroll
      for (int i = 1; i < WS; ++i, ++bdata)
        **bdata = prev = **bdata + prev;

      *g_ybar = prev;
    }

    {  // calculate vhat -----------------------
      float *bdata = s_block[tx];

      float prev = *bdata;
      ++bdata;

#pragma unroll
      for (int i = 1; i < WS; ++i, ++bdata)
        prev = *bdata + prev;

      *g_vhat = prev;
    }
  }
}

/**
 *  @ingroup gpu
 *  @brief Algorithm SAT stage 2
 *
 *  This function computes the algorithm stage S.2 following:
 *
 *  Sequentially for each \f$m\f$, but in parallel for each \f$n\f$,
 *  compute and store the \f$P_{m,n}(Y)\f$ and using the previously
 *  computed \f$P_{m,n}(\bar{Y})\f$.  Compute and store
 *  \f$s(P_{m,n}(Y))\f$.
 *
 *  @note The CUDA kernel functions (as this one) have many
 *  idiosyncrasies and should not be used lightly.
 *
 *  @see [Nehab:2011] cited in alg5() and figure in algSAT()
 *  @param[in,out] g_ybar All \f$P_{m,n}(\bar{Y})\f$ fixed to \f$P_{m,n}(Y)\f$
 *  @param[out] g_ysum All \f$s(P_{m,n}(Y))\f$
 */
__global__ void algSAT_stage2(const int c_n_size, const int c_m_size,
                              const int c_width, float *g_ybar, float *g_ysum)
{
  const int tx = threadIdx.x, ty = threadIdx.y, bx = blockIdx.x,
            col0 = bx * MW + ty, col = col0 * WS + tx;

  if (col >= c_width)
    return;

  g_ybar += col;
  float y = *g_ybar;
  int ln = HWS + tx;

  if (tx == WS - 1)
    g_ysum += col0;

  volatile __shared__ float s_block[MW][HWS + WS + 1];

  if (tx < HWS)
    s_block[ty][tx] = 0.f;
  else
    s_block[ty][ln] = 0.f;

  for (int n = 1; n < c_n_size; ++n)
  {
    // calculate ysum -----------------------
    s_block[ty][ln] = y;

    s_block[ty][ln] += s_block[ty][ln - 1];
    s_block[ty][ln] += s_block[ty][ln - 2];
    s_block[ty][ln] += s_block[ty][ln - 4];
    s_block[ty][ln] += s_block[ty][ln - 8];
    s_block[ty][ln] += s_block[ty][ln - 16];

    if (tx == WS - 1)
    {
      *g_ysum = s_block[ty][ln];
      g_ysum += c_m_size;
    }

    // fix ybar -> y -------------------------
    g_ybar += c_width;
    y = *g_ybar += y;
  }
}

/**
 *  @ingroup gpu
 *  @brief Algorithm SAT stage 3
 *
 *  This function computes the algorithm stage S.3 following:
 *
 *  Sequentially for each \f$n\f$, but in parallel for each \f$m\f$,
 *  compute and store the \f$P^T{m,n}(V)\f$ using the previously
 *  computed \f$P_{m-1,n}(Y)\f$, \f$P^T_{m,n}(\hat{V})\f$ and
 *  \f$s(P_{m,n}(Y))\f$.
 *
 *  @note The CUDA kernel functions (as this one) have many
 *  idiosyncrasies and should not be used lightly.
 *
 *  @see [Nehab:2011] cited in alg5() and figure in algSAT()
 *  @param[in] g_ysum All \f$s(P_{m,n}(Y))\f$
 *  @param[in,out] g_vhat All \f$P^T_{m,n}(\hat{V})\f$ fixed to
 *  \f$P^T_{m,n}(V)\f$
 */
__global__ void algSAT_stage3(const int c_m_size, const int c_height,
                              const float *g_ysum, float *g_vhat)
{
  const int tx = threadIdx.x, ty = threadIdx.y, by = blockIdx.y,
            row0 = by * MW + ty, row = row0 * WS + tx;

  if (row >= c_height)
    return;

  g_vhat += row;
  float y = 0.f, v = 0.f;

  if (row0 > 0)
    g_ysum += (row0 - 1) * c_m_size;

  for (int m = 0; m < c_m_size; ++m)
  {
    // fix vhat -> v -------------------------
    if (row0 > 0)
    {
      y = *g_ysum;
      g_ysum += 1;
    }

    v = *g_vhat += v + y;
    g_vhat += c_height;
  }
}

/**
 *  @ingroup gpu
 *  @brief Algorithm SAT stage 4
 *
 *  This function computes the algorithm stage S.4 following:
 *
 *  In parallel for all \f$m\f$ and \f$n\f$, compute \f$B_{m,n}(Y)\f$
 *  then compute and store \f$B_{m,n}(V)\f$ and using the previously
 *  computed \f$P_{m,n}(Y)\f$ and \f$P^T_{m,n}(V)\f$.
 *
 *  @note The CUDA kernel functions (as this one) have many
 *  idiosyncrasies and should not be used lightly.
 *
 *  @see [Nehab:2011] cited in alg5() and figure in algSAT()
 *  @param[in,out] g_inout The input and output image
 *  @param[in] g_y All \f$P_{m,n}(Y)\f$
 *  @param[in] g_v All \f$P^T_{m,n}(V)\f$
 */
__global__ void algSAT_stage4(const int c_width, const int c_height,
                              float *g_inout, const float *g_y,
                              const float *g_v)
{
  const int tx = threadIdx.x, ty = threadIdx.y, bx = blockIdx.x,
            by = blockIdx.y, col = bx * WS + tx, row0 = by * WS;

  __shared__ float s_block[WS][WS + 1];

  float(*bdata)[WS + 1] = (float(*)[WS + 1]) & s_block[ty][tx];

  g_inout += (row0 + ty) * c_width + col;
  if (by > 0)
    g_y += (by - 1) * c_width + col;
  if (bx > 0)
    g_v += (bx - 1) * c_height + row0 + tx;

#pragma unroll
  for (int i = 0; i < WS - (WS % SOW); i += SOW)
  {
    **bdata = *g_inout;
    bdata += SOW;
    g_inout += SOW * c_width;
  }
  if (ty < WS % SOW)
  {
    **bdata = *g_inout;
  }

  __syncthreads();

  if (ty == 0)
  {
    {  // calculate y -----------------------
      float(*bdata)[WS + 1] = (float(*)[WS + 1]) & s_block[0][tx];

      float prev;
      if (by > 0)
        prev = *g_y;
      else
        prev = 0.f;

#pragma unroll
      for (int i = 0; i < WS; ++i, ++bdata)
        **bdata = prev = **bdata + prev;
    }

    {  // calculate x -----------------------
      float *bdata = s_block[tx];

      float prev;
      if (bx > 0)
        prev = *g_v;
      else
        prev = 0.f;

#pragma unroll
      for (int i = 0; i < WS; ++i, ++bdata)
        *bdata = prev = *bdata + prev;
    }
  }

  __syncthreads();

  bdata = (float(*)[WS + 1]) & s_block[ty][tx];

  g_inout -= (WS - (WS % SOW)) * c_width;

#pragma unroll
  for (int i = 0; i < WS - (WS % SOW); i += SOW)
  {
    *g_inout = **bdata;
    bdata += SOW;
    g_inout += SOW * c_width;
  }
  if (ty < WS % SOW)
  {
    *g_inout = **bdata;
  }
}

/**
 *  @ingroup gpu
 *  @overload
 *  @brief Algorithm SAT stage 4 (not-in-place computation)
 *
 *  @note The CUDA kernel functions (as this one) have many
 *  idiosyncrasies and should not be used lightly.
 *
 *  @see [Nehab:2011] cited in alg5() and figure in algSAT()
 *  @param[out] g_out The output image
 *  @param[in] g_in The input image
 *  @param[in] g_y All \f$P_{m,n}(Y)\f$
 *  @param[in] g_v All \f$P^T_{m,n}(V)\f$
 */
__global__ void algSAT_stage4(const int c_width, const int c_height,
                              float *g_out, const float *g_in, const float *g_y,
                              const float *g_v)
{
  const int tx = threadIdx.x, ty = threadIdx.y, bx = blockIdx.x,
            by = blockIdx.y, col = bx * WS + tx, row0 = by * WS;

  __shared__ float s_block[WS][WS + 1];

  float(*bdata)[WS + 1] = (float(*)[WS + 1]) & s_block[ty][tx];

  g_in += (row0 + ty) * c_width + col;
  if (by > 0)
    g_y += (by - 1) * c_width + col;
  if (bx > 0)
    g_v += (bx - 1) * c_height + row0 + tx;

#pragma unroll
  for (int i = 0; i < WS - (WS % SOW); i += SOW)
  {
    **bdata = *g_in;
    bdata += SOW;
    g_in += SOW * c_width;
  }
  if (ty < WS % SOW)
  {
    **bdata = *g_in;
  }

  __syncthreads();

  if (ty == 0)
  {
    {  // calculate y -----------------------
      float(*bdata)[WS + 1] = (float(*)[WS + 1]) & s_block[0][tx];

      float prev;
      if (by > 0)
        prev = *g_y;
      else
        prev = 0.f;

#pragma unroll
      for (int i = 0; i < WS; ++i, ++bdata)
        **bdata = prev = **bdata + prev;
    }

    {  // calculate x -----------------------
      float *bdata = s_block[tx];

      float prev;
      if (bx > 0)
        prev = *g_v;
      else
        prev = 0.f;

#pragma unroll
      for (int i = 0; i < WS; ++i, ++bdata)
        *bdata = prev = *bdata + prev;
    }
  }

  __syncthreads();

  bdata = (float(*)[WS + 1]) & s_block[ty][tx];

  g_out += (row0 + ty) * c_width + col;

#pragma unroll
  for (int i = 0; i < WS - (WS % SOW); i += SOW)
  {
    *g_out = **bdata;
    bdata += SOW;
    g_out += SOW * c_width;
  }
  if (ty < WS % SOW)
  {
    *g_out = **bdata;
  }
}

/*
__host__
void prepare_algSAT(const int width
    dvector<float>& d_inout,
    dvector<float>& d_ybar,
    dvector<float>& d_vhat,
    dvector<float>& d_ysum,
    const float *h_in,
    const int& w,
    const int& h ) {

  algs.width = w;
  algs.height = h;

  if( w % 32 > 0 ) algs.width += (32 - (w % 32));
  if( h % 32 > 0 ) algs.height += (32 - (h % 32));

  calc_alg_setup( algs, algs.width, algs.height );
  up_alg_setup( algs );

  d_inout.copy_from( h_in, w, h, algs.width, algs.height );

  d_ybar.resize( algs.n_size * algs.width );
  d_vhat.resize( algs.m_size * algs.height );
  d_ysum.resize( algs.m_size * algs.n_size );

}
*/

__global__ void sat_init_kernel(int image_size, float xscale, float yscale,
                                float *thrustptr)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index = y * image_size + x;

  float texval = tex2D(textureReadDepth, x * xscale + 0.5f, y * yscale + 0.5f);

  thrustptr[index] = texval;
}

void resizeIfNecessary(thrust::device_vector<float> &vector, unsigned long size)
{
  if (vector.size() != size)
  {
    vector.resize(size);
  }
}

void callStages(thrust::device_vector<float> &inout,
                thrust::device_vector<float> &ybar,
                thrust::device_vector<float> &vhat,
                thrust::device_vector<float> &ysum, int computeWidth,
                int computeHeight, int compute_m_size, int compute_n_size)
{
  const int nWm = (computeWidth + MTS - 1) / MTS,
            nHm = (computeHeight + MTS - 1) / MTS;
  const dim3 cg_img(compute_m_size, compute_n_size);
  const dim3 cg_ybar(nWm, 1);
  const dim3 cg_vhat(1, nHm);

  float *d_inout = thrust::raw_pointer_cast(inout.data());
  float *d_ybar = thrust::raw_pointer_cast(ybar.data());
  float *d_vhat = thrust::raw_pointer_cast(vhat.data());
  float *d_ysum = thrust::raw_pointer_cast(ysum.data());

  algSAT_stage1<<<cg_img, dim3(WS, SOW)>>>(computeWidth, computeHeight,
      d_inout, d_ybar, d_vhat);
  algSAT_stage2<<<cg_ybar, dim3(WS, MW)>>>(compute_m_size, compute_n_size,
      computeWidth, d_ybar, d_ysum);
  algSAT_stage3<<<cg_vhat, dim3(WS, MW)>>>(compute_m_size, computeHeight,
      d_ysum, d_vhat);
  algSAT_stage4<<<cg_img, dim3(WS, SOW)>>>(computeWidth, computeHeight,
      d_inout, d_ybar, d_vhat);
}

void cudaSAT(cudaGraphicsResource_t &inputImage, int image_size,
             int screen_size_x, int screen_size_y, float z_threshold,
             thrust::device_vector<float> &inout,
             thrust::device_vector<float> &ybar,
             thrust::device_vector<float> &vhat,
             thrust::device_vector<float> &ysum)
{
  int computeWidth = image_size;
  int computeHeight = image_size;
  if (computeWidth % 32 > 0)
    computeWidth += (32 - (computeWidth % 32));
  if (computeHeight % 32 > 0)
    computeHeight += (32 - (computeHeight % 32));
  int compute_m_size = (computeWidth + WS - 1) / WS;
  int compute_n_size = (computeHeight + WS - 1) / WS;

  // set data structure sizes
  resizeIfNecessary(inout, computeWidth * computeHeight);
  resizeIfNecessary(ybar, compute_n_size * computeWidth);
  resizeIfNecessary(vhat, compute_m_size * computeHeight);
  resizeIfNecessary(ysum, compute_m_size * compute_n_size);


  // QElapsedTimer tm;
  // tm.start();

  // initialize occupancy function from inputImage
  textureReadDepth.normalized = 0;
  textureReadDepth.filterMode = cudaFilterModeLinear /*cudaFilterModePoint*/;
  textureReadDepth.addressMode[0] = cudaAddressModeWrap;
  textureReadDepth.addressMode[1] = cudaAddressModeWrap;

  cudaGraphicsMapResources(1, &inputImage);
  cudaArray_t input_array;
  cudaGraphicsSubResourceGetMappedArray(&input_array, inputImage, 0, 0);
  cudaChannelFormatDesc channeldesc;
  cudaGetChannelDesc(&channeldesc, input_array);

  cudaBindTextureToArray(&textureReadDepth, input_array, &channeldesc);

  dim3 dimBlock(64, 1, 1);
  dim3 dimGrid(divUp(image_size, dimBlock.x), divUp(image_size, dimBlock.y), 1);

  float *d_inout = thrust::raw_pointer_cast(inout.data());
  sat_init_kernel << <dimGrid, dimBlock>>>(image_size,
       static_cast<float>(screen_size_x) / static_cast<float>(image_size),
       static_cast<float>(screen_size_y) / static_cast<float>(image_size),
       d_inout);
  cudaThreadSynchronize();
  cudaUnbindTexture(&textureReadDepth);
  cudaGraphicsUnmapResources(1, &inputImage);

  callStages(inout, ybar, vhat, ysum, computeWidth, computeHeight,
             compute_m_size, compute_n_size);
}

/*
__host__
void algSAT( dvector<float>& d_out,
    dvector<float>& d_ybar,
    dvector<float>& d_vhat,
    dvector<float>& d_ysum,
    const dvector<float>& d_in,
    const alg_setup& algs ) {

  const int nWm = (algs.width+MTS-1)/MTS, nHm = (algs.height+MTS-1)/MTS;
  const dim3 cg_img( algs.m_size, algs.n_size );
  const dim3 cg_ybar( nWm, 1 );
  const dim3 cg_vhat( 1, nHm );

  algSAT_stage1<<< cg_img, dim3(WS, SOW) >>>( d_in, d_ybar, d_vhat );

  algSAT_stage2<<< cg_ybar, dim3(WS, MW) >>>( d_ybar, d_ysum );

  algSAT_stage3<<< cg_vhat, dim3(WS, MW) >>>( d_ysum, d_vhat );

  algSAT_stage4<<< cg_img, dim3(WS, SOW) >>>( d_out, d_in, d_ybar, d_vhat );

}

__host__
void algSAT( dvector<float>& d_inout,
    dvector<float>& d_ybar,
    dvector<float>& d_vhat,
    dvector<float>& d_ysum,
    const alg_setup& algs ) {

  const int nWm = (algs.width+MTS-1)/MTS, nHm = (algs.height+MTS-1)/MTS;
  const dim3 cg_img( algs.m_size, algs.n_size );
  const dim3 cg_ybar( nWm, 1 );
  const dim3 cg_vhat( 1, nHm );

  algSAT_stage1<<< cg_img, dim3(WS, SOW) >>>( d_inout, d_ybar, d_vhat );

  algSAT_stage2<<< cg_ybar, dim3(WS, MW) >>>( d_ybar, d_ysum );

  algSAT_stage3<<< cg_vhat, dim3(WS, MW) >>>( d_ysum, d_vhat );

  algSAT_stage4<<< cg_img, dim3(WS, SOW) >>>( d_inout, d_ybar, d_vhat );

}
*/

thrust::host_vector<float> algSAT(float *h_inout, int w, int h)
{
  int computeWidth = w;
  int computeHeight = w;
  if (computeWidth % 32 > 0)
    computeWidth += (32 - (computeWidth % 32));
  if (computeHeight % 32 > 0)
    computeHeight += (32 - (computeHeight % 32));
  int compute_m_size = (computeWidth + WS - 1) / WS;
  int compute_n_size = (computeHeight + WS - 1) / WS;

  // alg_setup algs;
  thrust::device_vector<float> inout(h_inout, h_inout + w * h);
  thrust::device_vector<float> ybar;
  thrust::device_vector<float> vhat;
  thrust::device_vector<float> ysum;

  // prepare_algSAT( algs, d_out, d_ybar, d_vhat, d_ysum, h_inout, w, h );
  resizeIfNecessary(inout, computeWidth * computeHeight);
  resizeIfNecessary(ybar, compute_n_size * computeWidth);
  resizeIfNecessary(vhat, compute_m_size * computeHeight);
  resizeIfNecessary(ysum, compute_m_size * compute_n_size);

  callStages(inout, ybar, vhat, ysum, computeWidth, computeHeight,
             compute_m_size, compute_n_size);

  cudaDeviceSynchronize();
  // algSAT(d_out, d_ybar, d_vhat, d_ysum, algs);

  // thrust::host_vector<float> result = inout;
  // thrust::host_vector<float> result(w * h);
  // thrust::copy(inout.begin(), inout.end(), result.begin());

  return inout;
}

namespace Placement
{

SummedAreaTable::SummedAreaTable(std::shared_ptr<CudaArrayProvider> inputImage)
  : inputImage(inputImage)
{
}

void SummedAreaTable::runKernel()
{
  int computeWidth = inputImage->getWidth();
  int computeHeight = inputImage->getHeight();
  if (computeWidth % 32 > 0)
    computeWidth += (32 - (computeWidth % 32));
  if (computeHeight % 32 > 0)
    computeHeight += (32 - (computeHeight % 32));
  int compute_m_size = (computeWidth + WS - 1) / WS;
  int compute_n_size = (computeHeight + WS - 1) / WS;

  std::cout << computeWidth << "/" << computeHeight << std::endl;

  // set data structure sizes
  resizeIfNecessary(inout, computeWidth * computeHeight);
  resizeIfNecessary(ybar, compute_n_size * computeWidth);
  resizeIfNecessary(vhat, compute_m_size * computeHeight);
  resizeIfNecessary(ysum, compute_m_size * compute_n_size);

  // initialize occupancy function from inputImage
  textureReadDepth.normalized = 0;
  textureReadDepth.filterMode = cudaFilterModeLinear /*cudaFilterModePoint*/;
  textureReadDepth.addressMode[0] = cudaAddressModeWrap;
  textureReadDepth.addressMode[1] = cudaAddressModeWrap;

  inputImage->map();
  cudaBindTextureToArray(textureReadDepth, inputImage->getArray(),
                         inputImage->getChannelDesc());

  dim3 dimBlock(64, 1, 1);
  dim3 dimGrid(divUp(inputImage->getWidth(), dimBlock.x),
               divUp(inputImage->getHeight(), dimBlock.y), 1);

  float *d_inout = thrust::raw_pointer_cast(inout.data());
  int image_size = inputImage->getWidth();
  int screen_size_x = inputImage->getWidth();
  int screen_size_y = inputImage->getHeight();

  sat_init_kernel<<<dimGrid, dimBlock>>>(image_size,
       static_cast<float>(screen_size_x) / static_cast<float>(image_size),
       static_cast<float>(screen_size_y) / static_cast<float>(image_size),
       d_inout);
  cudaThreadSynchronize();
  cudaUnbindTexture(&textureReadDepth);
  inputImage->unmap();

  callStages(inout, ybar, vhat, ysum, computeWidth, computeHeight,
             compute_m_size, compute_n_size);
}

thrust::device_vector<float> &SummedAreaTable::getResults()
{
  return inout;
}

}  // namespace Placement
