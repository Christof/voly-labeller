#ifndef SRC_TEXTURE_MAPPER_MANAGER_H_

#define SRC_TEXTURE_MAPPER_MANAGER_H_

#include <memory>
#include <vector>
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
class TextureMappersForLayer;

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

  void createTextureMappersForLayers(int layerCount);
  void
  initialize(Graphics::Gl *gl, std::shared_ptr<Graphics::FrameBufferObject> fbo,
             std::shared_ptr<ConstraintBufferObject> constraintBufferObject);

  void resize(int widht, int height);

  void update();

  void bindOccupancyTexture(int layerIndex);
  void bindDistanceTransform(int layerIndex);
  void bindApollonius(int layerIndex);

  std::shared_ptr<CudaTextureMapper> getOccupancyTextureMapper(int layerIndex);
  std::shared_ptr<CudaTextureMapper>
  getDistanceTransformTextureMapper(int layerIndex);
  std::shared_ptr<CudaTextureMapper> getApolloniusTextureMapper(int layerIndex);
  std::shared_ptr<CudaTextureMapper> getConstraintTextureMapper(int layerIndex);

  void cleanup();

  void saveOccupancy();
  void saveDistanceTransform();
  void saveApollonius();

  int getBufferSize();

 private:
  int bufferSize;
  std::vector<std::shared_ptr<TextureMappersForLayer>> mappersForLayers;
};

#endif  // SRC_TEXTURE_MAPPER_MANAGER_H_
