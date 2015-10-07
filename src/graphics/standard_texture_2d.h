#ifndef SRC_GRAPHICS_STANDARD_TEXTURE_2D_H_

#define SRC_GRAPHICS_STANDARD_TEXTURE_2D_H_

namespace Graphics
{

class Gl;

/**
 * \brief
 *
 *
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

 private:
  unsigned int texture;

  int width;
  int height;
  unsigned int format;

  Gl *gl;
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_STANDARD_TEXTURE_2D_H_
