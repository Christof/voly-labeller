#ifndef SRC_TEXTURE_MAPPERS_FOR_LAYER_H_

#define SRC_TEXTURE_MAPPERS_FOR_LAYER_H_

#include <memory>
#include "./graphics/frame_buffer_object.h"
#include "./graphics/standard_texture_2d.h"
#include "./graphics/gl.h"

class CudaTextureMapper;
namespace Placement
{
class Occlusion;
}

/**
 * \brief Container for CudaTextureMapper%s and corresponding textures for a
 * layer
 *
 * It also provides methods to save a texture to the filesystem as well
 * as methods to bind the textures to render them for debugging purposes.
 */
class TextureMappersForLayer
{
 public:
  TextureMappersForLayer(int bufferSize, int layerIndex);
  virtual ~TextureMappersForLayer();

  void initialize(Graphics::Gl *gl,
                  std::shared_ptr<Graphics::FrameBufferObject> fbo);

  void resize(int widht, int height);

  void update();

  void bindDistanceTransform();
  void bindApollonius();

  std::shared_ptr<CudaTextureMapper> getDistanceTransformTextureMapper();
  std::shared_ptr<CudaTextureMapper> getApolloniusTextureMapper();

  void cleanup();

  void saveDistanceTransform();
  void saveApollonius();

 private:
  std::shared_ptr<CudaTextureMapper> distanceTransformTextureMapper;
  std::shared_ptr<CudaTextureMapper> apolloniusTextureMapper;

  std::shared_ptr<Graphics::StandardTexture2d> distanceTransformTexture;
  std::shared_ptr<Graphics::StandardTexture2d> apolloniusTexture;

  int bufferSize;
  int layerIndex;
  int width;
  int height;

  bool saveDistanceTransformInNextFrame = false;
  bool saveApolloniusInNextFrame = false;

  void initializeMappers();
};

#endif  // SRC_TEXTURE_MAPPERS_FOR_LAYER_H_
