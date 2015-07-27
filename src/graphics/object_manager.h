#ifndef SRC_GRAPHICS_OBJECT_MANAGER_H_

#define SRC_GRAPHICS_OBJECT_MANAGER_H_

#include <Eigen/Core>
#include <memory>
#include <vector>
#include <map>
#include "./circular_buffer.h"
#include "./texture_address.h"

namespace Graphics
{

class BufferManager;
class TextureManager;
class ShaderManager;
class Gl;

struct ObjectData
{
  int primitiveType;
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
class ObjectManager
{
 public:
  explicit ObjectManager(std::shared_ptr<TextureManager> textureManager,
      std::shared_ptr<ShaderManager> shaderManager);
  virtual ~ObjectManager();

  void initialize(Gl *gl, uint maxObjectCount, uint bufferSize);

  int addObject(const std::vector<float> &vertices,
                const std::vector<float> &normals,
                const std::vector<float> &colors,
                const std::vector<float> &texCoords,
                const std::vector<uint> &indices,
                int primitiveType = GL_TRIANGLES);
  bool removeObject(int objID);

  void render();

  void renderImmediately(int objectId);
  void renderLater(int objectId, Eigen::Matrix4f transform);
  void renderLater(ObjectData object);

  bool setObjectTexture(int objectId, uint textureId);
  bool setObjectTransform(int objectId, const Eigen::Matrix4f &transform);

 private:
  const GLbitfield MAP_FLAGS = GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT;
  const GLbitfield CREATE_FLAGS = MAP_FLAGS | GL_DYNAMIC_STORAGE_BIT;

  int objectCount = 0;
  std::map<int, ObjectData> objects;

  std::shared_ptr<BufferManager> bufferManager;
  std::shared_ptr<TextureManager> textureManager;
  std::shared_ptr<ShaderManager> shaderManager;

  CircularBuffer<float[16]> transformBuffer;
  CircularBuffer<TextureAddress> textureAddressBuffer;
  CircularBuffer<DrawElementsIndirectCommand> commandsBuffer;

  Gl *gl;

  std::vector<ObjectData> objectsForFrame;

  void renderObjects(std::vector<ObjectData> objects);
  DrawElementsIndirectCommand createDrawCommand(const ObjectData &objectData,
                                                int counter);
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_OBJECT_MANAGER_H_
