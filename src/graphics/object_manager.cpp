#include "./object_manager.h"
#include <QLoggingCategory>
#include <vector>
#include <map>
#include <string>
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
    commandsBuffer(GL_DRAW_INDIRECT_BUFFER)
{
  int customBufferCount = 2;
  for (int customBufferIndex = 0; customBufferIndex < customBufferCount;
       ++customBufferIndex)
  {
    customBuffers.push_back(
        std::make_shared<ShaderBuffer>(GL_SHADER_STORAGE_BUFFER));
  }
}

ObjectManager::~ObjectManager()
{
  qCInfo(omChan) << "Destructor of ObjectManager";
}

void ObjectManager::initialize(Gl *gl, uint maxObjectCount, uint bufferSize)
{
  this->gl = gl;

  bufferManager->initialize(gl, maxObjectCount, bufferSize);

  commandsBuffer.initialize(gl, 3 * maxObjectCount, CREATE_FLAGS, MAP_FLAGS);
  transformBuffer.initialize(gl, 3 * maxObjectCount, CREATE_FLAGS, MAP_FLAGS);
  for (auto customBuffer : customBuffers)
    customBuffer->initialize(gl, 24 * maxObjectCount, CREATE_FLAGS, MAP_FLAGS);
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

  return ObjectData(nextFreeId++, bufferInformation.vertexBufferOffset,
                    bufferInformation.indexBufferOffset,
                    static_cast<int>(indices.size()), shaderProgramId,
                    primitiveType);
}

ObjectData ObjectManager::cloneForDifferentShader(const ObjectData &object,
                                                  int shaderProgramId)
{
  return ObjectData(nextFreeId++, object.getVertexOffset(),
                    object.getIndexOffset(), object.getIndexSize(),
                    shaderProgramId, object.getPrimitiveType());
}

ObjectData ObjectManager::clone(const ObjectData &object)
{
  return ObjectData(nextFreeId++, object.getVertexOffset(),
                    object.getIndexOffset(), object.getIndexSize(),
                    object.getShaderProgramId(), object.getPrimitiveType());
}

void ObjectManager::render(const RenderData &renderData)
{
  std::map<int, std::vector<ObjectData>> objectsByShader;
  for (auto &object : objectsForFrame)
  {
    objectsByShader[object.getShaderProgramId()].push_back(object);
  }

  for (auto &shaderObjectPair : objectsByShader)
  {
    qCDebug(omChan) << "Render objects with shader" << shaderObjectPair.first
                    << "with" << shaderObjectPair.second.size() << "objects";

    shaderManager->bindForHABuffer(shaderObjectPair.first, renderData);

    std::map<int, std::vector<ObjectData>> objectsByPrimitiveType;
    for (auto &object : shaderObjectPair.second)
    {
      objectsByPrimitiveType[object.getPrimitiveType()].push_back(object);
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
  uint objectCount = static_cast<uint>(objects.size());
  DrawElementsIndirectCommand *commands = commandsBuffer.reserve(objectCount);
  auto *matrices = transformBuffer.reserve(objectCount);

  int customBufferIndex = 0;
  std::vector<CustomBufferData> customBuffersData;
  for (auto customBuffer : customBuffers)
  {
    int customBufferSize =
        objects[0].getCustomBufferSize(customBufferIndex) * objectCount;

    if (customBufferSize)
    {
      void *custom = customBuffer->reserve(customBufferSize);
      qCDebug(omChan) << "Buffer" << customBuffer.get() << "Index"
                      << customBufferIndex << "customBufferSize"
                      << customBufferSize << "custom" << custom << "matrices"
                      << matrices;
      customBuffersData.push_back({.pointer = custom,
                                   .size = customBufferSize,
                                   .index = customBufferIndex });
    }

    ++customBufferIndex;
  }

  int counter = 0;
  for (auto &objectData : objects)
  {
    commands[counter] = createDrawCommand(objectData, counter);

    auto *transform = &matrices[counter];
    auto modelMatrix = objectData.modelMatrix;

    // encode objectId into modelMatrix
    int objectId = objectData.getId();
    modelMatrix(3, 0) = *reinterpret_cast<float *>(&objectId);

    memcpy(transform, modelMatrix.data(), sizeof(float[16]));

    if (objectData.hasCustomBuffer())
    {
      for (auto &bufferData : customBuffersData)
        objectData.fillBufferElementFor(bufferData.index, bufferData.pointer,
                                        counter);
    }

    ++counter;
  }

  transformBuffer.bindBufferRange(0, objectCount);
  for (auto &bufferData : customBuffersData)
  {
    customBuffers[bufferData.index]->bindBufferRange(
        bufferData.index + 1, bufferData.size);
  }

  // We didn't use MAP_COHERENT here - make sure data is on the gpu
  // glAssert(gl->glMemoryBarrier(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT));
  glAssert(gl->glMemoryBarrier(GL_ALL_BARRIER_BITS));

  // draw
  qCDebug(omChan, "head: %ld headoffset %p objectcount: %u",
          commandsBuffer.getHead(), commandsBuffer.headOffset(), objectCount);

  glAssert(gl->glMultiDrawElementsIndirect(
      objects[0].getPrimitiveType(), GL_UNSIGNED_INT,
      commandsBuffer.headOffset(), objectCount, 0));

  bufferManager->unbind();

  commandsBuffer.onUsageComplete(objectCount);
  transformBuffer.onUsageComplete(objectCount);
  for (auto &bufferData : customBuffersData)
  {
    customBuffers[bufferData.index]->onUsageComplete(bufferData.size);
  }
}

ObjectManager::DrawElementsIndirectCommand
ObjectManager::createDrawCommand(const ObjectData &objectData, int counter)
{
  DrawElementsIndirectCommand command;

  command.count = objectData.getIndexSize();
  command.instanceCount = 1;
  command.firstIndex = objectData.getIndexOffset();
  command.baseVertex = objectData.getVertexOffset();
  command.baseInstance = counter;

  qCDebug(omChan, "counter: %d count: %u firstIndex: %u baseVertex: %u",
          counter, command.count, command.firstIndex, command.baseVertex);

  return command;
}

}  // namespace Graphics
