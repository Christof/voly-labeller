#ifndef SRC_GRAPHICS_VERTEX_ARRAY_H_

#define SRC_GRAPHICS_VERTEX_ARRAY_H_

#include <vector>
#include "./vertex_buffer.h"
#include "./gl.h"

namespace Graphics
{

/**
 * \brief Container for vertex array for a defined GL primitive used in
 * rendering
 *
 * Basic container used for rendering. Manages multiple
 * streams (adjacent vertex elements) and binds each
 * stream to a buffer object.
 *
 * @see VertexBuffer
 */
class VertexArray
{
 public:
  VertexArray(Gl *gl, GLenum primitiveMode, int positionElementsCount = 3);
  ~VertexArray();

  /**
   * \brief Draws all vertices contained in data member
   *
   * Binds each buffer to a vertex attrib array depending on the different
   * shaders defined in Scene
   *
   */
  void draw();

  void addStream(std::vector<float> stream, int elementSize = 3);
  void addStream(size_t count, int elementSize = 3);
  void updateStream(size_t index, std::vector<float> stream);

 private:
  Gl *gl;
  std::vector<VertexBuffer *> data;
  GLenum primitiveMode;
  GLuint vertexArrayId;
  int positionElementsCount;
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_VERTEX_ARRAY_H_
