#include "./object_manager.h"
#include <QLoggingCategory>
#include "./gl.h"
#include "./buffer_manager.h"
#include "./texture_manager.h"

namespace Graphics
{

QLoggingCategory omChan("Graphics.ObjectManager");

// TODO(SIR): remove
ObjectManager *ObjectManager::instance = nullptr;

ObjectManager::ObjectManager(std::shared_ptr<TextureManager> textureManager)
  : bufferManager(std::make_shared<BufferManager>()),
    textureManager(textureManager), transformBuffer(GL_SHADER_STORAGE_BUFFER),
    textureAddressBuffer(GL_SHADER_STORAGE_BUFFER),
    commandsBuffer(GL_DRAW_INDIRECT_BUFFER)

{
  // TODO(SIR): remove
  instance = this;
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
  textureAddressBuffer.initialize(gl, 3 * maxObjectCount, CREATE_FLAGS,
                                  MAP_FLAGS);
}

int ObjectManager::addObject(const std::vector<float> &vertices,
                             const std::vector<float> &normals,
                             const std::vector<float> &colors,
                             const std::vector<float> &texCoords,
                             const std::vector<uint> &indices)
{
  auto bufferInformation =
      bufferManager->addObject(vertices, normals, colors, texCoords, indices);

  ObjectData object;
  object.vertexOffset = bufferInformation.vertexBufferOffset;
  object.vertexSize = bufferInformation.vertexCount;
  object.indexOffset = bufferInformation.indexBufferOffset;
  object.indexSize = indices.size();
  object.textureAddress = { 0, 0.0f, 0, { 1.0f, 1.0f } };
  object.transform = Eigen::Matrix4f::Identity();

  int objectId = objectCount++;

  objects.insert(std::make_pair(objectId, object));

  return objectId;
}

bool ObjectManager::setObjectTexture(int objectId, uint textureId)
{
  if (!objects.count(objectId))
    return false;

  objects[objectId].textureAddress = textureManager->getAddressFor(textureId);
  qCDebug(
      omChan,
      "VolySceneManager::setObjectTexture: objID:%d handle: %lu slice: %f\n",
      objectId, objects[objectId].textureAddress.containerHandle,
      objects[objectId].textureAddress.texPage);

  return true;
}

bool ObjectManager::setObjectTransform(int objectId,
                                       const Eigen::Matrix4f &transform)
{
  if (objects.count(objectId) == 0)
    return false;

  objects[objectId].transform = transform;

  return true;
}

void ObjectManager::render()
{
  bufferManager->bind();

  // prepare per object buffers
  uint objectCount = objects.size();
  DrawElementsIndirectCommand *commands = commandsBuffer.reserve(objectCount);
  auto *matrices = transformBuffer.reserve(objectCount);
  TextureAddress *textures = textureAddressBuffer.reserve(objectCount);

  int counter = 0;
  for (auto objectIterator = objects.begin(); objectIterator != objects.end();
       ++objectIterator, ++counter)
  {
    auto objectData = objectIterator->second;

    DrawElementsIndirectCommand *command = &commands[counter];
    command->count = objectData.indexSize;
    command->instanceCount = 1;
    command->firstIndex = objectData.indexOffset;
    command->baseVertex = objectData.vertexOffset;
    command->baseInstance = counter;

    auto *transform = &matrices[counter];
    memcpy(transform, objectData.transform.data(),
           sizeof(float[16]));

    TextureAddress *texaddr = &textures[counter];
    *texaddr = objectData.textureAddress;

    qCDebug(omChan, "counter: %d count: %u firstIndex: %u baseVertex: %u",
            counter, command->count, command->firstIndex, command->baseVertex);
    qCDebug(omChan, "counter: %d handle: %lu slice: %f", counter,
            texaddr->containerHandle, texaddr->texPage);
  }

  qCDebug(omChan, "objectcount: %u/%ld", objectCount, commandsBuffer.size());

  int mapRange = objectCount;

  mapRange = std::min(128, ((mapRange / 4) + 1) * 4);
  transformBuffer.bindBufferRange(0, mapRange);
  textureAddressBuffer.bindBufferRange(1, mapRange);

  // We didn't use MAP_COHERENT here - make sure data is on the gpu
  glAssert(gl->glMemoryBarrier(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT));

  // draw
  qCDebug(omChan, "head: %ld headoffset %p objectcount: %un",
          commandsBuffer.getHead(), commandsBuffer.headOffset(), objectCount);

  glAssert(gl->glMultiDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT,
                                           commandsBuffer.headOffset(),
                                           objectCount, 0));

  bufferManager->unbind();

  commandsBuffer.onUsageComplete(mapRange);
  transformBuffer.onUsageComplete(mapRange);
  textureAddressBuffer.onUsageComplete(mapRange);
}
}  // namespace Graphics
