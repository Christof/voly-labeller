#ifndef SRC_GRAPHICS_STANDARD_TEXTURE_2D_H_

#define SRC_GRAPHICS_STANDARD_TEXTURE_2D_H_

#include <string>

namespace Graphics
{

class Gl;

/**
 * \brief Encapsulates a plain old OpenGL texture
 *
 * This classes is inteded for use with textures which are used
 * for post processing. Otherwise the bindless textures should be used.
 */
class StandardTexture2d
{
 public:
  StandardTexture2d(int width, int height, unsigned int format);
  virtual ~StandardTexture2d();

  void initialize(Gl *gl);

  void bind();
  void unbind();

  unsigned int getId();

  void save(std::string filename);

  int getWidth();
  int getHeight();

 private:
  unsigned int texture;

  int width;
  int height;
  unsigned int format;

  Gl *gl;

  int getComponentsPerPixel();
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_STANDARD_TEXTURE_2D_H_
