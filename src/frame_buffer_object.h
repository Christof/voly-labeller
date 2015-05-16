#ifndef SRC_FRAME_BUFFER_OBJECT_H_

#define SRC_FRAME_BUFFER_OBJECT_H_

#include <memory>

class Gl;

/**
 * \brief Encapsulates a frame buffer with textures for color and depth
 * attachments
 *
 * FrameBufferObject::initialize must be called once, and FrameBufferObject::resize
 * if the window is resized (mind that it must be called from the rendering thread).
 *
 * The draw into the frame buffer call FrameBufferObject::bind. After drawing
 * call FrameBufferObject::unbind.
 *
 * Using FrameBufferObject::bindColorTexture or FrameBufferObject::bindDepthTexture
 * the corresponding textures can be used.
 */
class FrameBufferObject
{
 public:
  FrameBufferObject() = default;
  ~FrameBufferObject();

  void initialize(Gl *gl, int width, int height);

  void resize(Gl *gl, int width, int height);

  void bind();
  void unbind();

  void bindColorTexture(unsigned int textureUnit);
  void bindDepthTexture(unsigned int textureUnit);

 private:
  unsigned int framebuffer = 0;
  unsigned int renderTexture = 0;
  unsigned int depthTexture = 0;
  Gl *gl;

  void resizeTexture(int texture, int width, int height, unsigned int component,
                     unsigned int format);
};

#endif  // SRC_FRAME_BUFFER_OBJECT_H_
