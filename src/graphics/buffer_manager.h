#ifndef SRC_GRAPHICS_BUFFER_MANAGER_H_

#define SRC_GRAPHICS_BUFFER_MANAGER_H_

#include <Eigen/Core>
#include <vector>
#include <map>
#include <memory>
#include "./buffer_hole_manager.h"
#include "./attribute_buffer.h"

namespace Graphics
{

class Gl;

struct BufferInformation
{
  uint vertexBufferOffset;
  uint indexBufferOffset;
};

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
  void initialize(Gl *gl, uint maxObjectCount, uint bufferSize);

  BufferInformation addObject(const std::vector<float> &vertices,
                              const std::vector<float> &normals,
                              const std::vector<float> &colors,
                              const std::vector<float> &texCoords,
                              const std::vector<uint> &indices);

  void bind();
  void unbind();

 private:
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

  void initializeDrawIdBuffer(uint maxObjectCount);
};
}  // namespace Graphics

#endif  // SRC_GRAPHICS_BUFFER_MANAGER_H_
