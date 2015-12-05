#ifndef SRC_CONSTRAINT_BUFFER_OBJECT_H_

#define SRC_CONSTRAINT_BUFFER_OBJECT_H_

#include "./graphics/gl.h"

/**
 * \brief Frame buffer for constraints mask
 *
 * It only contains a color attachment of type unsigned byte.
 * ConstraintBufferObject::initialize must be called once before using the
 * buffer.
 *
 * To draw into the frame buffer call ConstraintBufferObject::bind. After drawing
 * call ConstraintBufferObject::unbind.
 *
 * Using ConstraintBufferObject::bindTexture binds the color texture.
 */
class ConstraintBufferObject
{
 public:
  ConstraintBufferObject() = default;
  virtual ~ConstraintBufferObject();

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

#endif  // SRC_CONSTRAINT_BUFFER_OBJECT_H_
