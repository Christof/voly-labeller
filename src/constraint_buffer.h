#ifndef SRC_CONSTRAINT_BUFFER_H_

#define SRC_CONSTRAINT_BUFFER_H_

#include "./graphics/gl.h"

/**
 * \brief
 *
 *
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

 private:
  Graphics::Gl *gl;
  unsigned int framebuffer = 0;
  unsigned int renderTexture = 0;

  void resizeAndSetColorAttachment(int width, int height);
  void resizeTexture(int texture, int width, int height, unsigned int component,
                     unsigned int format, unsigned int type);
};

#endif  // SRC_CONSTRAINT_BUFFER_H_