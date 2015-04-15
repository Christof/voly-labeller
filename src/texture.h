#ifndef SRC_TEXTURE_H_

#define SRC_TEXTURE_H_

#include <string>
#include "./gl.h"

/**
 * \brief Loads an image and provides functions to use it as a texture
 *
 * The image is loaded automatically in the constructor and can
 * be used by calling Texture::Bind.
 */
class Texture
{
 public:
  explicit Texture(std::string filename);
  virtual ~Texture();

  void initialize(Gl *gl);

  /**
   * \brief Binds the texture to the given texture unit
   */
  void bind(Gl *gl, GLenum textureUnit) const;
  int getWidth() const;
  int getHeight() const;

 private:
  const GLenum textureTarget = GL_TEXTURE_2D;
  std::string filename;
  GLuint texture;
  int width;
  int height;
};

#endif  // SRC_TEXTURE_H_
