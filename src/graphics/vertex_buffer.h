#ifndef SRC_GRAPHICS_VERTEX_BUFFER_H_

#define SRC_GRAPHICS_VERTEX_BUFFER_H_

#include <vector>
#include "./gl.h"

namespace Graphics
{

/**
 * \brief OpenGL Buffer for float data
 *
 * Automatically generates and binds a new OpenGL Buffer Object
 * for the defined data vector
 *
 * @see VertexArray
 */
class VertexBuffer
{
 public:
  VertexBuffer(Gl *gl, std::vector<float> data, int elementSize)
    : gl(gl), elementSize(elementSize)
  {
    gl->glGenBuffers(1, &id);
    gl->glBindBuffer(GL_ARRAY_BUFFER, id);
    gl->glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(float), &data[0],
                     GL_STATIC_DRAW);
    size = data.size();
  }

  VertexBuffer(Gl *gl, int elementCount, int elementSize)
    : gl(gl), size(elementCount), elementSize(elementSize)
  {
    gl->glGenBuffers(1, &id);
    gl->glBindBuffer(GL_ARRAY_BUFFER, id);
    gl->glBufferData(GL_ARRAY_BUFFER, elementCount * sizeof(float), nullptr,
                     GL_DYNAMIC_DRAW);
  }

  ~VertexBuffer()
  {
    gl->glBindBuffer(GL_ARRAY_BUFFER, 0);
    gl->glDeleteBuffers(1, &id);
  }

  void bind()
  {
    gl->glBindBuffer(GL_ARRAY_BUFFER, id);
  }

  void update(std::vector<float> data)
  {
    bind();
    gl->glBufferSubData(GL_ARRAY_BUFFER, 0, data.size() * sizeof(float),
                        &data[0]);
    size = data.size();
  }

  size_t getSize()
  {
    return size;
  }

  int getElementSize()
  {
    return elementSize;
  }

 private:
  Gl *gl;
  GLuint id;

  size_t size;
  int elementSize;
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_VERTEX_BUFFER_H_
