#include "./standard_texture_2d.h"
#include <string>
#include <vector>
#include <algorithm>
#include "./gl.h"
#include "../utils/image_persister.h"

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

void StandardTexture2d::save(std::string filename)
{
  int componets = getComponentsPerPixel();
  int pixelCount = width * height;
  std::vector<float> pixels(pixelCount * componets);

  int usedBuffer = 0;
  gl->glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &usedBuffer);

  unsigned int fboId = 0;
  gl->glGenFramebuffers(1, &fboId);
  gl->glBindFramebuffer(GL_FRAMEBUFFER, fboId);
  gl->glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             GL_TEXTURE_2D, texture, 0);

  if (format == GL_R32F)
  {
    gl->glReadPixels(0, 0, width, height, GL_RED, GL_FLOAT, pixels.data());
    ::ImagePersister::saveR32F(pixels.data(), width, height, filename);

    convertToR8IAndSave(pixels, filename);
  }
  else if (format == GL_RGBA32F)
  {
    gl->glReadPixels(0, 0, width, height, GL_RGBA, GL_FLOAT, pixels.data());
    ::ImagePersister::saveRGBA32F(pixels.data(), width, height, filename);
  }

  gl->glBindFramebuffer(GL_FRAMEBUFFER, usedBuffer);
  gl->glDeleteBuffers(1, &fboId);
}

int StandardTexture2d::getWidth()
{
  return width;
}

int StandardTexture2d::getHeight()
{
  return height;
}

int StandardTexture2d::getComponentsPerPixel()
{
  switch (format)
  {
  case GL_R32F:
    return 1;
  case GL_RGBA32F:
    return 4;
  default:
    throw std::runtime_error("Format '" + std::to_string(format) +
                             "' not implemented");
  }
}

void StandardTexture2d::convertToR8IAndSave(const std::vector<float> &pixels,
                                            std::string filename)
{
  auto minmaxElement = std::minmax_element(pixels.begin(), pixels.end());
  float maxValue = *minmaxElement.second;
  std::vector<unsigned char> charPixels(pixels.size());
  std::transform(pixels.begin(), pixels.end(), charPixels.begin(),
                 [maxValue](float pixel)
                 {
    return static_cast<unsigned char>(pixel / maxValue * 255);
  });

  int index = filename.find_last_of(".");
  std::string filenameWithoutExtension = filename.substr(0, index);
  std::string name = filenameWithoutExtension + ".png";
  ::ImagePersister::saveR8I(charPixels.data(), width, height, name);
}

}  // namespace Graphics
