#ifndef SRC_GRAPHICS_ATTRIBUTE_BUFFER_H_

#define SRC_GRAPHICS_ATTRIBUTE_BUFFER_H_

#include "./gl.h"
#include <cassert>

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
  GLuint m_glID;
  uint m_Size;
  GLuint m_type;
  static const GLuint c_buf_type = GL_ARRAY_BUFFER;

 public:
  AttributeBuffer(int count, int componentCount);
  virtual ~AttributeBuffer();

  bool isInitialized() const;
  GLuint getId() const;

  /*
  GLuint type() const
  {
    return m_type;
  }
  uint primitiveSize() const
  {
    return Size;
  };
  uint size() const
  {
    return m_Size;
  }
  */
  uint sizeBytes() const;

  void initialize(Gl *gl, uint size, GLuint type = c_buf_type);
  // void copy(AttributeBuffer const &buffer);

  // void resize(uint size);

  void bindAttrib(int attribnum);
  void bindAttribDivisor(int attribnum, int divisor);
  void bind();
  void unbind();

  template <typename T> void setData(std::vector<T> values, uint offset)
  {
    assert(m_glID != 0);
    assert(sizeof(T) == elementSize());
    assert(offset + (values.size() / componentCount) <= m_Size);

    const uint byteSize =
        values.size() * componentSize;  // no Size, because vector is flat
    const uint byteOffset = offset * elementSize();

    glAssert(gl->getDirectStateAccess()->glNamedBufferSubDataEXT(
        m_glID, byteOffset, byteSize, values.data()));
  }

  template <typename T>
  void getData(std::vector<T> &values, uint offset, uint size)
  {
    assert(m_glID != 0);
    assert(sizeof(T) == elementSize());

    values.resize(m_Size);
    if (size == 0)
      size = m_Size;

    glAssert(gl->getDirectStateAccess()->glGetNamedBufferSubDataEXT(
        m_glID, offset * elementSize(), size * elementSize(), values.data()));
  }

 private:
  const int componentCount;
  const int componentSize;
  Gl *gl;

  int elementSize() const;
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_ATTRIBUTE_BUFFER_H_
