#ifndef SRC_GRAPHICS_OBJECT_MANAGER_H_

#define SRC_GRAPHICS_OBJECT_MANAGER_H_

#include <Eigen/Core>
#include <memory>
#include <vector>
#include <map>
#include <functional>
#include "./circular_buffer.h"
#include "./shader_buffer.h"
#include "./texture_address.h"
#include "./render_data.h"

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

  int shaderProgramId;

  int customBufferSize;
  std::function<void(void*)> setBuffer;

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

  ObjectData addObject(const std::vector<float> &vertices,
                const std::vector<float> &normals,
                const std::vector<float> &colors,
                const std::vector<float> &texCoords,
                const std::vector<uint> &indices, int shaderProgramId,
                int primitiveType = GL_TRIANGLES);
  int addShader(std::string vertexShaderPath, std::string fragmentShaderPath);
  int addTexture(std::string path);
  TextureAddress getAddressFor(int textureId);
  bool removeObject(int objID);

  void render(const RenderData &renderData);

  void renderImmediately(ObjectData objectData);
  void renderLater(ObjectData object);

 private:
  const GLbitfield MAP_FLAGS = GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT;
  const GLbitfield CREATE_FLAGS = MAP_FLAGS | GL_DYNAMIC_STORAGE_BIT;

  std::shared_ptr<BufferManager> bufferManager;
  std::shared_ptr<TextureManager> textureManager;
  std::shared_ptr<ShaderManager> shaderManager;

  CircularBuffer<float[16]> transformBuffer;
  ShaderBuffer customBuffer;
  CircularBuffer<DrawElementsIndirectCommand> commandsBuffer;

  Gl *gl;

  std::vector<ObjectData> objectsForFrame;

  void renderObjects(std::vector<ObjectData> objects);
  DrawElementsIndirectCommand createDrawCommand(const ObjectData &objectData,
                                                int counter);
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_OBJECT_MANAGER_H_
