#ifndef SRC_GRAPHICS_FRAME_BUFFER_OBJECT_H_

#define SRC_GRAPHICS_FRAME_BUFFER_OBJECT_H_

#include <memory>
#include <vector>

namespace Graphics
{

class Gl;

/**
 * \brief Encapsulates a frame buffer with textures for color and depth
 * attachments
 *
 * FrameBufferObject::initialize must be called once, and
 * FrameBufferObject::resize if the window is resized (mind that it must be
 * called from the rendering thread).
 *
 * To draw into the frame buffer call FrameBufferObject::bind. After drawing
 * call FrameBufferObject::unbind.
 *
 * Using FrameBufferObject::bindColorTexture or
 * FrameBufferObject::bindDepthTexture binds
 * the corresponding texture so that it can be used.
 */
class FrameBufferObject
{
 public:
  explicit FrameBufferObject(unsigned int layerCount);
  ~FrameBufferObject();

  void initialize(Gl *gl, int width, int height);

  void resize(int width, int height);

  void bind();
  void unbind();

  void bindColorTexture(int index, unsigned int textureUnit);
  void bindAccumulatedLayersTexture(unsigned int textureUnit);
  void bindDepthTexture(unsigned int textureUnit);

  unsigned int getColorTextureId(int index);
  unsigned int getAccumulatedLayersTextureId();
  unsigned int getDepthTextureId();

  int getLayerCount();

 private:
  unsigned int layerCount;
  unsigned int framebuffer = 0;
  unsigned int colorTexturesArray = 0;
  unsigned int depthTexture = 0;
  unsigned int accumulatedLayersTexture = 0;
  Gl *gl;

  void resizeAndSetColorAttachment(unsigned int texture, int attachment,
                                   int width, int height);
  void resizeAndSetColorArrayAttachment(unsigned int texture,
                                        unsigned int attachment, int width,
                                        int height);
  void resizeAndSetDepthAttachment(int width, int height);
  void resizeTexture(unsigned int texture, int width, int height,
                     unsigned int component, unsigned int format,
                     unsigned int type);
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_FRAME_BUFFER_OBJECT_H_
