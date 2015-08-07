#ifndef SRC_GRAPHICS_ATTRIBUTE_BUFFER_H_

#define SRC_GRAPHICS_ATTRIBUTE_BUFFER_H_

#include <cassert>
#include <vector>
#include "./gl.h"

namespace Graphics
{
/**
 * \brief
 *
 *
 */
class AttributeBuffer
{
 protected:
  GLuint id = 0;
  uint count = 0;
  GLuint bufferType;
  static const GLuint DEFAULT_BUFFER_TYPE = GL_ARRAY_BUFFER;

 public:
  AttributeBuffer(int componentCount, int componentSize, GLenum componentType);
  virtual ~AttributeBuffer();

  bool isInitialized() const;
  GLuint getId() const;
  int getComponentCount() const;

  /*
  GLuint type() const
  {
    return bufferType;
  }
  uint primitiveSize() const
  {
    return Size;
  };
  uint size() const
  {
    return count;
  }
  */
  uint sizeBytes() const;

  void initialize(Gl *gl, uint size, GLuint type = DEFAULT_BUFFER_TYPE);

  void bindAttrib(int attribnum);
  void bindAttribDivisor(int attribnum, int divisor);
  void bind();
  void unbind();

  template <typename T> void setData(std::vector<T> values, uint offset = 0)
  {
    assert(id != 0);
    assert(sizeof(T) == componentSize);
    assert(offset + (values.size() / componentCount) <= count);

    const uint byteSize =
        values.size() * componentSize;  // no Size, because vector is flat
    const uint byteOffset = offset * elementSize();

    glAssert(gl->glNamedBufferSubData(id, byteOffset, byteSize, values.data()));
  }

  template <typename T>
  void getData(std::vector<T> &values, uint offset = 0, uint size = 0)
  {
    assert(id != 0);
    assert(sizeof(T) == componentSize);

    values.resize(count);
    if (size == 0)
      size = count;

    glAssert(gl->glGetNamedBufferSubData(id, offset * elementSize(),
                                         size * elementSize(), values.data()));
  }

 private:
  const int componentCount;
  const int componentSize;
  const GLenum componentType;
  Gl *gl;

  int elementSize() const;
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_ATTRIBUTE_BUFFER_H_
