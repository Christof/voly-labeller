#ifndef SRC_TEXTURE_MAPPER_MANAGER_H_

#define SRC_TEXTURE_MAPPER_MANAGER_H_

#include <memory>
#include "./graphics/frame_buffer_object.h"
#include "./graphics/standard_texture_2d.h"
#include "./graphics/gl.h"

class CudaTextureMapper;
class Occupancy;
class DistanceTransform;

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

  void initialize(Graphics::Gl *gl,
                  std::shared_ptr<Graphics::FrameBufferObject> fbo);

  void resize(int widht, int height);

  void update();

  void bindOccupancyTexture();
  void bindDistanceTransform();

  std::shared_ptr<CudaTextureMapper> getOccupancyTextureMapper();
  std::shared_ptr<CudaTextureMapper> getDistanceTransformTextureMapper();

  void cleanup();

 private:
  std::shared_ptr<CudaTextureMapper> colorTextureMapper;
  std::shared_ptr<CudaTextureMapper> positionsTextureMapper;
  std::shared_ptr<CudaTextureMapper> distanceTransformTextureMapper;
  std::shared_ptr<CudaTextureMapper> occupancyTextureMapper;
  std::shared_ptr<Graphics::StandardTexture2d> occupancyTexture;
  std::shared_ptr<Graphics::StandardTexture2d> distanceTransformTexture;

  std::unique_ptr<Occupancy> occupancy;
  std::unique_ptr<DistanceTransform> distanceTransform;
  int bufferSize;
  int width;
  int height;

  void initializeMappers(std::shared_ptr<Graphics::FrameBufferObject> fbo);
};

#endif  // SRC_TEXTURE_MAPPER_MANAGER_H_
