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

/**
 * \brief Stores offsets into index and vertex element buffers
 */
struct BufferInformation
{
  uint vertexBufferOffset;
  uint indexBufferOffset;
};

/**
 * \brief Holds all buffers and provides methods to add new objects
 *
 * The vertex data consists of:
 * - position (3 dimensions)
 * - normal (3 dimensions)
 * - color (4 dimensions; red, green, blue and alpha)
 * - texture coordinates (2 dimensions)
 * - internally also the draw id (1 dimension)
 *
 * Also indices are store separately.
 *
 * New objects can be add by calling #addObject().
 * 
 * All buffers can be bound with #bind() for usage
 * and unbound with #unbind() afterwards.
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
