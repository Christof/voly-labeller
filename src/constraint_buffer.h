#ifndef SRC_CONSTRAINT_BUFFER_H_

#define SRC_CONSTRAINT_BUFFER_H_

#include "./graphics/gl.h"

/**
 * \brief Frame buffer for constraints
 *
 * It only contains a color attachment of type unsigned byte.
 * ConstraintBuffer::initialize must be called once, and
 * ConstraintBuffer::resize if the window is resized (mind that it must be
 * called from the rendering thread).
 *
 * The draw into the frame buffer call FrameBufferObject::bind. After drawing
 * call FrameBufferObject::unbind.
 *
 * Using ConstraintBuffer::bindTexture binds the color texture.
 */
class ConstraintBuffer
{
 public:
  ConstraintBuffer() = default;
  virtual ~ConstraintBuffer();

  void initialize(Graphics::Gl *gl, int width, int height);

  void bind();
  void unbind();

  void bindTexture(unsigned int textureUnit);
  unsigned int getRenderTextureId();

  int getWidth();
  int getHeight();

 private:
  Graphics::Gl *gl;
  unsigned int framebuffer = 0;
  unsigned int renderTexture = 0;
  int width;
  int height;

  void resizeAndSetColorAttachment(int width, int height);
  void resizeTexture(int texture, int width, int height, unsigned int component,
                     unsigned int format, unsigned int type);
};

#endif  // SRC_CONSTRAINT_BUFFER_H_
