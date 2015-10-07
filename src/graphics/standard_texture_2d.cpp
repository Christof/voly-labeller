#include "./standard_texture_2d.h"
#include "./gl.h"

namespace Graphics
{

StandardTexture2d::StandardTexture2d(int width, int height, unsigned int format)
  : width(width), height(height), format(format)
{
}

StandardTexture2d::~StandardTexture2d()
{
  gl->glDeleteTextures(1, &texture);
}

void StandardTexture2d::initialize(Gl *gl)
{
  this->gl = gl;

  gl->glCreateTextures(GL_TEXTURE_2D, 1, &texture);
  gl->glTextureStorage2D(texture, 1, format, width, height);

  gl->glTextureParameterf(texture, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  gl->glTextureParameterf(texture, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  gl->glTextureParameteri(texture, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
  gl->glTextureParameteri(texture, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
}
void StandardTexture2d::bind()
{
  gl->glBindTexture(GL_TEXTURE_2D, texture);
}

void StandardTexture2d::unbind()
{
  gl->glBindTexture(GL_TEXTURE_2D, 0);
}

unsigned int StandardTexture2d::getId()
{
  return texture;
}

}  // namespace Graphics
