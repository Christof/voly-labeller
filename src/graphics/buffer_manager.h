#ifndef SRC_GRAPHICS_BUFFER_MANAGER_H_

#define SRC_GRAPHICS_BUFFER_MANAGER_H_

#include <Eigen/Core>
#include <vector>
#include <map>
#include <memory>
#include "./gl.h"
#include "./buffer_hole_manager.h"
#include "./circular_buffer.h"
#include "./attribute_buffer.h"
#include "./texture_address.h"

namespace Graphics
{

class TextureManager;

struct ObjectData
{
  int vertexOffset;
  int indexOffset;
  int vertexSize;
  int indexSize;

  TextureAddress textureAddress;

  Eigen::Matrix4f transform;
};

struct DrawElementsIndirectCommand
{
  GLuint count;
  GLuint instanceCount;
  GLuint firstIndex;
  GLuint baseVertex;
  GLuint baseInstance;
};

/**
 * \brief
 *
 *
 */
class BufferManager
{
 public:
  BufferManager(std::shared_ptr<TextureManager> textureManager);
  virtual ~BufferManager();
  void initialize(Gl *gl, uint maxObjectCount, uint bufferSize);

  int addObject(const std::vector<float> &vertices,
                const std::vector<float> &normals,
                const std::vector<float> &colors,
                const std::vector<float> &texCoords,
                const std::vector<uint> &indices);
  bool removeObject(int objID);

  void render();

  bool setObjectTexture(int objectId, uint textureId);

 private:
  const GLbitfield MAP_FLAGS = GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT;
  const GLbitfield CREATE_FLAGS = MAP_FLAGS | GL_DYNAMIC_STORAGE_BIT;

  std::shared_ptr<TextureManager> textureManager;
  int objectCount = 0;
  GLuint vertexArrayId;
  Gl *gl;
  std::map<int, ObjectData> objects;

  AttributeBuffer positionBuffer;
  AttributeBuffer normalBuffer;
  AttributeBuffer colorBuffer;
  AttributeBuffer texCoordBuffer;
  AttributeBuffer drawIdBuffer;
  AttributeBuffer indexBuffer;

  BufferHoleManager vertexBufferManager;
  BufferHoleManager indexBufferManager;

  CircularBuffer<float[16]> transformBuffer;
  CircularBuffer<TextureAddress> textureAddressBuffer;
  CircularBuffer<DrawElementsIndirectCommand> commandsBuffer;
};
}  // namespace Graphics

#endif  // SRC_GRAPHICS_BUFFER_MANAGER_H_
