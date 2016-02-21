#ifndef SRC_TEXTURE_MAPPERS_FOR_LAYER_H_

#define SRC_TEXTURE_MAPPERS_FOR_LAYER_H_

#include <memory>
#include "./graphics/frame_buffer_object.h"
#include "./graphics/standard_texture_2d.h"
#include "./graphics/gl.h"

class CudaTextureMapper;
namespace Placement
{
class Occupancy;
class DistanceTransform;
}
class ConstraintBufferObject;

/**
 * \brief
 *
 *
 */
class TextureMappersForLayer
{
 public:
  explicit TextureMappersForLayer(int bufferSize);
  virtual ~TextureMappersForLayer();

  void
  initialize(Graphics::Gl *gl, std::shared_ptr<Graphics::FrameBufferObject> fbo,
             std::shared_ptr<ConstraintBufferObject> constraintBufferObject);

  void resize(int widht, int height);

  void update();

  void bindOccupancyTexture();
  void bindDistanceTransform();
  void bindApollonius();

  std::shared_ptr<CudaTextureMapper> getOccupancyTextureMapper();
  std::shared_ptr<CudaTextureMapper> getDistanceTransformTextureMapper();
  std::shared_ptr<CudaTextureMapper> getApolloniusTextureMapper();
  std::shared_ptr<CudaTextureMapper> getConstraintTextureMapper();

  void cleanup();

  void saveOccupancy();
  void saveDistanceTransform();
  void saveApollonius();

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

  std::unique_ptr<Placement::Occupancy> occupancy;
  std::unique_ptr<Placement::DistanceTransform> distanceTransform;
  int bufferSize;
  int width;
  int height;

  bool saveOccupancyInNextFrame = false;
  bool saveDistanceTransformInNextFrame = false;
  bool saveApolloniusInNextFrame = false;

  void initializeMappers(
      std::shared_ptr<Graphics::FrameBufferObject> fbo,
      std::shared_ptr<ConstraintBufferObject> constraintBufferObject);
};

#endif  // SRC_TEXTURE_MAPPERS_FOR_LAYER_H_
