#ifndef SRC_TEXTURE_MAPPER_MANAGER_H_

#define SRC_TEXTURE_MAPPER_MANAGER_H_

#include <memory>
#include "./graphics/frame_buffer_object.h"
#include "./graphics/standard_texture_2d.h"
#include "./graphics/gl.h"

class CudaTextureMapper;

/**
 * \brief
 *
 *
 */
class TextureMapperManager
{
 public:
  TextureMapperManager(int bufferSize);
  ~TextureMapperManager();

  void
  initialize(Graphics::Gl *gl,
             std::shared_ptr<Graphics::FrameBufferObject> frameBufferObject);

  void resize(int widht, int height);

  void update();

  void cleanup();

 private:
  std::shared_ptr<CudaTextureMapper> colorTextureMapper;
  std::shared_ptr<CudaTextureMapper> positionsTextureMapper;
  std::shared_ptr<CudaTextureMapper> distanceTransformTextureMapper;
  std::shared_ptr<CudaTextureMapper> occupancyTextureMapper;
  std::shared_ptr<Graphics::StandardTexture2d> occupancyTexture;
  std::shared_ptr<Graphics::StandardTexture2d> distanceTransformTexture;

  int bufferSize;
  int width;
  int height;
};

#endif  // SRC_TEXTURE_MAPPER_MANAGER_H_
