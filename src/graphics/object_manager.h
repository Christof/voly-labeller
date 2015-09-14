#ifndef SRC_GRAPHICS_OBJECT_MANAGER_H_

#define SRC_GRAPHICS_OBJECT_MANAGER_H_

#include <memory>
#include <vector>
#include <map>
#include <string>
#include "./circular_buffer.h"
#include "./shader_buffer.h"
#include "./texture_address.h"
#include "./render_data.h"
#include "./object_data.h"

namespace Graphics
{

class BufferManager;
class TextureManager;
class ShaderManager;
class Gl;

/**
 * \brief Provides means to create and render objects
 *
 * Objects are created either with the ObjectManager::addObject method
 * or by cloning another ObjectData instance with ObjectManager::clone.
 * In both cases an ObjectData instance is returned, which internally stores
 * offsets into buffers, the used shader and other data about the object.
 *
 * To get an object rendered, the ObjectManager::renderLater method must be
 * called, which keeps the given object in a list.
 *
 * All objects added with ObjectManager::renderLater are render when the
 * ObjectManager::render method is called by the HABuffer.
 *
 * The HABuffer itself as well as post-processing steps use the
 * ObjectManager::renderImediately method to render a provided object
 * directly without storing it in the list and waiting for the
 * ObjectManager::render call.
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
  ObjectData cloneForDifferentShader(const ObjectData &object,
                                     int shaderProgramId);
  ObjectData clone(const ObjectData &object);
  bool removeObject(int objID);

  void render(const RenderData &renderData);

  void renderImmediately(ObjectData objectData);
  void renderLater(ObjectData object);

 private:
  struct DrawElementsIndirectCommand
  {
    GLuint count;
    GLuint instanceCount;
    GLuint firstIndex;
    GLuint baseVertex;
    GLuint baseInstance;
  };

  int nextFreeId = 1;

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
