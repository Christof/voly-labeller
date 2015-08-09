#include "./object_manager.h"
#include <QLoggingCategory>
#include <vector>
#include <map>
#include <cassert>
#include <algorithm>
#include "./gl.h"
#include "./buffer_manager.h"
#include "./texture_manager.h"
#include "./shader_manager.h"

namespace Graphics
{

QLoggingCategory omChan("Graphics.ObjectManager");

ObjectManager::ObjectManager(std::shared_ptr<TextureManager> textureManager,
                             std::shared_ptr<ShaderManager> shaderManager)
  : bufferManager(std::make_shared<BufferManager>()),
    textureManager(textureManager), shaderManager(shaderManager),
    transformBuffer(GL_SHADER_STORAGE_BUFFER),
    customBuffer(GL_SHADER_STORAGE_BUFFER),
    commandsBuffer(GL_DRAW_INDIRECT_BUFFER)
{
}

ObjectManager::~ObjectManager()
{
}

void ObjectManager::initialize(Gl *gl, uint maxObjectCount, uint bufferSize)
{
  this->gl = gl;

  bufferManager->initialize(gl, maxObjectCount, bufferSize);

  commandsBuffer.initialize(gl, 3 * maxObjectCount, CREATE_FLAGS, MAP_FLAGS);
  transformBuffer.initialize(gl, 3 * maxObjectCount, CREATE_FLAGS, MAP_FLAGS);
  customBuffer.initialize(gl, 12 * maxObjectCount, CREATE_FLAGS, MAP_FLAGS);
}

ObjectData ObjectManager::addObject(const std::vector<float> &vertices,
                                    const std::vector<float> &normals,
                                    const std::vector<float> &colors,
                                    const std::vector<float> &texCoords,
                                    const std::vector<uint> &indices,
                                    int shaderProgramId, int primitiveType)
{
  auto bufferInformation =
      bufferManager->addObject(vertices, normals, colors, texCoords, indices);

  ObjectData object;
  object.vertexOffset = bufferInformation.vertexBufferOffset;
  object.indexOffset = bufferInformation.indexBufferOffset;
  object.indexSize = indices.size();
  object.customBufferSize = 0;
  object.setBuffer = nullptr;
  object.transform = Eigen::Matrix4f::Identity();
  object.shaderProgramId = shaderProgramId;
  object.primitiveType = primitiveType;

  return object;
}

int ObjectManager::addShader(std::string vertexShaderPath,
                             std::string fragmentShaderPath)
{
  auto id = shaderManager->addShader(vertexShaderPath, fragmentShaderPath);

  qCDebug(omChan) << "Added" << vertexShaderPath.c_str() << "|"
                  << fragmentShaderPath.c_str() << "which got id" << id;

  return id;
}

int ObjectManager::addTexture(std::string path)
{
  return textureManager->addTexture(path);
}

TextureAddress ObjectManager::getAddressFor(int textureId)
{
  return textureManager->getAddressFor(textureId);
}

void ObjectManager::render(const RenderData &renderData)
{
  std::map<int, std::vector<ObjectData>> objectsByShader;
  for (auto &object : objectsForFrame)
  {
    objectsByShader[object.shaderProgramId].push_back(object);
  }

  for (auto &shaderObjectPair : objectsByShader)
  {
    qCDebug(omChan) << "Render objects with shader" << shaderObjectPair.first
                    << "with" << shaderObjectPair.second.size() << "objects";

    shaderManager->bind(shaderObjectPair.first, renderData);

    std::map<int, std::vector<ObjectData>> objectsByPrimitiveType;
    for (auto &object : shaderObjectPair.second)
    {
      objectsByPrimitiveType[object.primitiveType].push_back(object);
    }

    for (auto pair : objectsByPrimitiveType)
    {
      renderObjects(objectsByPrimitiveType[pair.first]);
    }
  }

  objectsForFrame.clear();
}

void ObjectManager::renderImmediately(ObjectData objectData)
{
  std::vector<ObjectData> objs = { objectData };

  renderObjects(objs);
}

void ObjectManager::renderLater(ObjectData object)
{
  objectsForFrame.push_back(object);
}

void ObjectManager::renderObjects(std::vector<ObjectData> objects)
{
  bufferManager->bind();

  // prepare per object buffers
  uint objectCount = objects.size();
  DrawElementsIndirectCommand *commands = commandsBuffer.reserve(objectCount);
  auto *matrices = transformBuffer.reserve(objectCount);
  int customBufferSize = objects[0].customBufferSize * objectCount;

  void *custom = nullptr;
  if (customBufferSize)
    custom = customBuffer.reserve(customBufferSize);

  qCDebug(omChan) << "customBufferSize" << customBufferSize << "custom"
                  << custom << "matrices" << matrices;

  int counter = 0;
  for (auto &objectData : objects)
  {
    commands[counter] = createDrawCommand(objectData, counter);

    auto *transform = &matrices[counter];
    memcpy(transform, objectData.transform.data(), sizeof(float[16]));

    if (objectData.setBuffer && customBufferSize)
    {
      assert(objectData.customBufferSize != 0);
      objectData.setBuffer(static_cast<char *>(custom) +
                           counter * objectData.customBufferSize);
    }

    ++counter;
  }

  transformBuffer.bindBufferRange(0, objectCount);
  if (customBufferSize)
    customBuffer.bindBufferRange(1, customBufferSize);

  // We didn't use MAP_COHERENT here - make sure data is on the gpu
  // glAssert(gl->glMemoryBarrier(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT));
  glAssert(gl->glMemoryBarrier(GL_ALL_BARRIER_BITS));

  // draw
  qCDebug(omChan, "head: %ld headoffset %p objectcount: %u",
          commandsBuffer.getHead(), commandsBuffer.headOffset(), objectCount);

  glAssert(gl->glMultiDrawElementsIndirect(
      objects[0].primitiveType, GL_UNSIGNED_INT, commandsBuffer.headOffset(),
      objectCount, 0));

  bufferManager->unbind();

  commandsBuffer.onUsageComplete(objectCount);
  transformBuffer.onUsageComplete(objectCount);
  if (customBufferSize)
    customBuffer.onUsageComplete(customBufferSize);
}

DrawElementsIndirectCommand
ObjectManager::createDrawCommand(const ObjectData &objectData, int counter)
{
  DrawElementsIndirectCommand command;

  command.count = objectData.indexSize;
  command.instanceCount = 1;
  command.firstIndex = objectData.indexOffset;
  command.baseVertex = objectData.vertexOffset;
  command.baseInstance = counter;

  qCDebug(omChan, "counter: %d count: %u firstIndex: %u baseVertex: %u",
          counter, command.count, command.firstIndex, command.baseVertex);

  return command;
}

}  // namespace Graphics
