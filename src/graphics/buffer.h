#ifndef SRC_GRAPHICS_BUFFER_H_

#define SRC_GRAPHICS_BUFFER_H_

#include "./gl.h"

namespace Graphics
{
/**
 * \brief
 *
 *
 */
class Buffer
{
 public:
  Buffer();
  Buffer(Gl *gl, uint size);
  virtual ~Buffer();

  Buffer(Buffer const &buffer);

  GLuint getId() const;
  GLuint getType() const;
  uint getSize() const;
  GLuint64 getGpuPointer() const;
  bool isInitialized() const;

  void adopt(Buffer const &buffer);
  void forget();

  void copy(Buffer const &buffer);

  void initialize(Gl *gl, uint size);
  void resize(uint size);
  void terminate();

  void clear(uint value);
  void clear(int value);
  void clear(float value);
  void setData(const void *raw, uint byteCount, uint offset = 0);
  void getData(void *raw, uint byteCount, uint offset = 0);

 protected:
  Gl *gl;
  GLuint id;
  GLuint64 gpuPointer;
  uint size;
  static const GLuint bufferType = GL_TEXTURE_BUFFER;
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_BUFFER_H_
