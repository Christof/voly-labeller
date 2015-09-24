#ifndef SRC_PLACEMENT_CUDA_TEXTURE_MAPPER_H_

#define SRC_PLACEMENT_CUDA_TEXTURE_MAPPER_H_

struct CUgraphicsResource_st;

/**
 * \brief
 *
 *
 */
class CudaTextureMapper
{
 public:
  CudaTextureMapper(unsigned int textureId);
  virtual ~CudaTextureMapper();

 private:
  unsigned int textureId;
  CUgraphicsResource_st *resource = 0;
};

#endif  // SRC_PLACEMENT_CUDA_TEXTURE_MAPPER_H_
