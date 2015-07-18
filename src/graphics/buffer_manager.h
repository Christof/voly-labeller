#ifndef SRC_GRAPHICS_BUFFER_MANAGER_H_

#define SRC_GRAPHICS_BUFFER_MANAGER_H_

#include "./gl.h"
#include "./buffer_hole_manager.h"
#include "./attribute_buffer.h"

namespace Graphics
{
/**
 * \brief
 *
 *
 */
class BufferManager
{
 public:
  BufferManager();
  virtual ~BufferManager();
  void initialize(Gl *gl, uint maxobjects, uint buffersize);

  int addObject(const std::vector<float> &vertices,
                const std::vector<float> &normals,
                const std::vector<float> &colors,
                const std::vector<float> &texCoords,
                const std::vector<uint> &indices);
  bool removeObject(int objID);

 private:
  int objectCount = 0;
  GLuint vertexArrayId;
  Gl *gl;

  AttributeBuffer positionBuffer;
  AttributeBuffer normalBuffer;
  AttributeBuffer colorBuffer;
  AttributeBuffer texCoordBuffer;
  AttributeBuffer drawIdBuffer;
  AttributeBuffer indexBuffer;

  BufferHoleManager vertexBufferManager;
  BufferHoleManager indexBufferManager;
};
}  // namespace Graphics

#endif  // SRC_GRAPHICS_BUFFER_MANAGER_H_
