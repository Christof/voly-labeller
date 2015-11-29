#ifndef SRC_TEXTURE_MAPPER_MANAGER_H_

#define SRC_TEXTURE_MAPPER_MANAGER_H_

#include <memory>
#include "./graphics/frame_buffer_object.h"
#include "./graphics/standard_texture_2d.h"
#include "./graphics/gl.h"

class CudaTextureMapper;
class Occupancy;
class DistanceTransform;
class ConstraintBuffer;

/**
 * \brief Container for all CudaTextureMapper%s and corresponding textures
 *
 * It also provides methods to save a texture to the filesystem as well
 * as methods to bind the textures to render them for debugging purposes.
 */
class TextureMapperManager
{
 public:
  explicit TextureMapperManager(int bufferSize);
  ~TextureMapperManager();

  void initialize(Graphics::Gl *gl,
                  std::shared_ptr<Graphics::FrameBufferObject> fbo,
                  std::shared_ptr<ConstraintBuffer> constraintBuffer);

  void resize(int widht, int height);

  void update();

  void bindOccupancyTexture();
  void bindDistanceTransform();
  void bindApollonius();

  std::shared_ptr<CudaTextureMapper> getOccupancyTextureMapper();
  std::shared_ptr<CudaTextureMapper> getDistanceTransformTextureMapper();
  std::shared_ptr<CudaTextureMapper> getApolloniusTextureMapper();

  void cleanup();

  void saveOccupancy();
  void saveDistanceTransform();
  void saveApollonius();

  int getBufferSize();

 private:
  std::shared_ptr<CudaTextureMapper> colorTextureMapper;
  std::shared_ptr<CudaTextureMapper> positionsTextureMapper;
  std::shared_ptr<CudaTextureMapper> distanceTransformTextureMapper;
  std::shared_ptr<CudaTextureMapper> occupancyTextureMapper;
  std::shared_ptr<CudaTextureMapper> apolloniusTextureMapper;
  std::shared_ptr<CudaTextureMapper> constraintTextureMapper;

  std::shared_ptr<Graphics::StandardTexture2d> occupancyTexture;
  std::shared_ptr<Graphics::StandardTexture2d> distanceTransformTexture;
  std::shared_ptr<Graphics::StandardTexture2d> apolloniusTexture;

  std::unique_ptr<Occupancy> occupancy;
  std::unique_ptr<DistanceTransform> distanceTransform;
  int bufferSize;
  int width;
  int height;

  bool saveOccupancyInNextFrame = false;
  bool saveDistanceTransformInNextFrame = false;
  bool saveApolloniusInNextFrame = false;

  void initializeMappers(std::shared_ptr<Graphics::FrameBufferObject> fbo,
                         std::shared_ptr<ConstraintBuffer> constraintBuffer);
};

#endif  // SRC_TEXTURE_MAPPER_MANAGER_H_
