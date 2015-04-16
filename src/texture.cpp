#include "./texture.h"
#include <string>
#include <QImage>
#include <QDebug>

Texture::Texture(std::string filename) : filename(filename)
{
}

Texture::~Texture()
{
  glBindTexture(textureTarget, 0);
  glDeleteTextures(1, &texture);
}

void Texture::initialize(Gl *gl)
{
  try
  {
    QImage image(filename.c_str());

    width = image.width();
    height = image.height();

    // glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    // glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);

    glAssert(gl->glGenTextures(1, &texture));
    glAssert(gl->glBindTexture(textureTarget, texture));
    glAssert(gl->glTexImage2D(textureTarget, 0, GL_RGBA, width, height, 0,
                              GL_BGRA, GL_UNSIGNED_BYTE, image.bits()));

    glTexParameterf(textureTarget, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(textureTarget, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(textureTarget, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(textureTarget, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glAssert(gl->glBindTexture(textureTarget, 0));
  }
  catch (std::exception &error)
  {
    qCritical() << "Error loading texture '" << filename.c_str()
             << "': " << error.what();
    throw;
  }
}

void Texture::bind(Gl *gl, GLenum textureUnit) const
{
  glAssert(gl->glActiveTexture(textureUnit));
  glAssert(gl->glBindTexture(textureTarget, texture));
}

int Texture::getWidth() const
{
  return width;
}

int Texture::getHeight() const
{
  return height;
}

