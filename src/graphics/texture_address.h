#ifndef SRC_GRAPHICS_TEXTURE_ADDRESS_H_

#define SRC_GRAPHICS_TEXTURE_ADDRESS_H_

#include <QOpenGLFunctions>

namespace Graphics
{

struct TextureAddress
{
  GLuint64 containerHandle;
  GLfloat texPage;
  GLint reserved;
  GLfloat texscale[2];
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_TEXTURE_ADDRESS_H_
