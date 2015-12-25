#ifndef SRC_GRAPHICS_FRAME_BUFFER_OBJECT_H_

#define SRC_GRAPHICS_FRAME_BUFFER_OBJECT_H_

#include <memory>

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
  FrameBufferObject() = default;
  ~FrameBufferObject();

  void initialize(Gl *gl, int width, int height);

  void resize(int width, int height);

  void bind();
  void unbind();

  void bindColorTexture(unsigned int textureUnit);
  void bindPositionTexture(unsigned int textureUnit);
  void bindDepthTexture(unsigned int textureUnit);
  void bindColorTexture2(unsigned int textureUnit);
  void bindPositionTexture2(unsigned int textureUnit);

  unsigned int getRenderTextureId();
  unsigned int getPositionTextureId();
  unsigned int getDepthTextureId();

 private:
  unsigned int framebuffer = 0;
  unsigned int renderTexture = 0;
  unsigned int positionTexture = 0;
  unsigned int depthTexture = 0;
  unsigned int positionTexture2 = 0;
  unsigned int renderTexture2 = 0;
  Gl *gl;

  void resizeAndSetColorAttachment(int texture, int attachment, int width,
                                   int height);
  void resizeAndSetPositionAttachment(int texture, int attachment, int width,
                                      int height);
  void resizeAndSetDepthAttachment(int width, int height);
  void resizeTexture(int texture, int width, int height, unsigned int component,
                     unsigned int format, unsigned int type);
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_FRAME_BUFFER_OBJECT_H_
