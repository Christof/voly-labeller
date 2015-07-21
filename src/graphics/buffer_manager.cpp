#include "./buffer_manager.h"
#include <vector>
#include "./texture_manager.h"

namespace Graphics
{

BufferManager::BufferManager(std::shared_ptr<TextureManager> textureManager)
  : textureManager(textureManager), positionBuffer(3, sizeof(float), GL_FLOAT),
    normalBuffer(3, sizeof(float), GL_FLOAT),
    colorBuffer(4, sizeof(float), GL_FLOAT),
    texCoordBuffer(2, sizeof(float), GL_FLOAT),
    drawIdBuffer(1, sizeof(uint), GL_UNSIGNED_INT),
    indexBuffer(1, sizeof(uint), GL_UNSIGNED_INT), vertexBufferManager(0),
    indexBufferManager(0), commandsBuffer(GL_DRAW_INDIRECT_BUFFER)
{
}

BufferManager::~BufferManager()
{
  if (!gl)
    return;

  glAssert(gl->glDeleteVertexArrays(1, &vertexArrayId));

  /*
  // clear textures
  for (auto texture = m_Textures.begin(); texture != m_Textures.end();
  texture++)
  {
    delete (*texture);
    *texture = nullptr;
  }
  m_Textures.clear();

  m_TextureManager.Shutdown();
  */
}

void BufferManager::initialize(Gl *gl, uint maxObjectCount, uint bufferSize)
{
  this->gl = gl;
  // gl_assert(m_VertexArrayID == 0);

  // m_TextureManager.Init(true, 8);

  vertexBufferManager = BufferHoleManager(bufferSize);
  indexBufferManager = BufferHoleManager(bufferSize);

  glAssert(gl->glGenVertexArrays(1, &vertexArrayId));
  glAssert(gl->glBindVertexArray(vertexArrayId));

  positionBuffer.initialize(gl, bufferSize);
  normalBuffer.initialize(gl, bufferSize);
  colorBuffer.initialize(gl, bufferSize);
  texCoordBuffer.initialize(gl, bufferSize);

  indexBuffer.initialize(gl, bufferSize, GL_ELEMENT_ARRAY_BUFFER);

  // initialize DrawID buffer - ascending ids
  drawIdBuffer.initialize(gl, maxObjectCount);
  std::vector<uint> drawids;
  uint idval = 0;
  drawids.resize(maxObjectCount);

  for_each(drawids.begin(), drawids.end(),
           [&idval](std::vector<uint>::value_type &v)
           {
    v = idval++;
  });
  // std::cout << "drawIDs:" << drawids << std::endl;

  drawIdBuffer.setData(drawids);

  // bind per vertex attibutes to vertex array
  positionBuffer.bindAttrib(0);
  normalBuffer.bindAttrib(1);
  colorBuffer.bindAttrib(2);
  texCoordBuffer.bindAttrib(3);
  drawIdBuffer.bindAttribDivisor(4, 1);

  glAssert(gl->glBindVertexArray(0));

  commandsBuffer.initialize(gl, 3 * maxObjectCount, CREATE_FLAGS, MAP_FLAGS);
  // transformBuffer.init(3 * maxObjectCount, CREATE_FLAGS, MAP_FLAGS);
  // texAddressBuffer.init(3 * maxObjectCount, CREATE_FLAGS, MAP_FLAGS);
}

int BufferManager::addObject(const std::vector<float> &vertices,
                             const std::vector<float> &normals,
                             const std::vector<float> &colors,
                             const std::vector<float> &texCoords,
                             const std::vector<uint> &indices)
{
  assert(vertices.size() /
             static_cast<float>(positionBuffer.getComponentCount()) ==
         normals.size() / static_cast<float>(normalBuffer.getComponentCount()));
  assert(vertices.size() /
             static_cast<float>(positionBuffer.getComponentCount()) ==
         colors.size() / static_cast<float>(colorBuffer.getComponentCount()));
  assert(vertices.size() /
             static_cast<float>(positionBuffer.getComponentCount()) ==
         texCoords.size() /
             static_cast<float>(texCoordBuffer.getComponentCount()));

  const uint vertexCount = vertices.size() / positionBuffer.getComponentCount();

  // try to reserve buffer storage for objects

  uint vertexBufferOffset;
  uint indexBufferOffset;

  bool reserve_success =
      vertexBufferManager.reserve(vertexCount, vertexBufferOffset);
  if (reserve_success)
  {
    reserve_success =
        indexBufferManager.reserve(indices.size(), indexBufferOffset);
    if (!reserve_success)
    {
      vertexBufferManager.release(vertexBufferOffset);
    }
  }

  if (!reserve_success)
    return -1;

  // fill buffers
  positionBuffer.setData(vertices, vertexBufferOffset);
  normalBuffer.setData(normals, vertexBufferOffset);
  colorBuffer.setData(colors, vertexBufferOffset);
  texCoordBuffer.setData(texCoords, vertexBufferOffset);

  indexBuffer.setData(indices, indexBufferOffset);

  ObjectData object;
  object.vertexOffset = vertexBufferOffset;
  object.vertexSize = vertexCount;
  object.indexOffset = indexBufferOffset;
  object.indexSize = indices.size();
  object.textureAddress = { 0, 0.0f, 0, { 1.0f, 1.0f } };
  object.transform = Eigen::Matrix4f::Identity();

  int objectId = objectCount++;

  objects.insert(std::make_pair(objectId, object));

  return objectId;
}

bool BufferManager::setObjectTexture(int objectId, uint textureId)
{
  if (!objects.count(objectId))
    return false;

  objects[objectId].textureAddress = textureManager->getAddressFor(textureId);
  qDebug("VolySceneManager::setObjectTexture: objID:%d handle: %lu slice: %f\n",
         objectId, objects[objectId].textureAddress.containerHandle,
         objects[objectId].textureAddress.texPage);

  return true;
}

void BufferManager::render()
{
  glAssert(gl->glBindVertexArray(vertexArrayId));

  // prepare per object buffers
  uint objectCount = objects.size();
  DrawElementsIndirectCommand *commands = commandsBuffer.reserve(objectCount);
  /*
  GLMatrix * matrices = m_TransformBuffer.reserve(objectcount);
  TexAddress * textures = m_TexAddressBuffer.reserve(objectcount);
  */

  int counter = 0;
  for (auto objectIterator = objects.begin(); objectIterator != objects.end();
       ++objectIterator, ++counter)
  {
    DrawElementsIndirectCommand *command = &commands[counter];
    command->count = objectIterator->second.indexSize;
    command->instanceCount = 1;
    command->firstIndex = objectIterator->second.indexOffset;
    command->baseVertex = objectIterator->second.vertexOffset;
    command->baseInstance = counter;

    /*
    GLMatrix *transform = &matrices[counter];
    memcpy(transform, objit->second.m_transform.data(), sizeof(GLMatrix));
    //std::cout << "matrix:"
    //          << objit->second.m_transform << std::endl;

    TexAddress *texaddr = &textures[counter];
    *texaddr = objit->second.m_TextureAddress;
    //texaddr->m_container_handle = objit->second.m_TextureAddress;
    //texaddr->m_tex_page = 0.0f;


    */
    printf("counter: %d count: %u firstIndex: %u baseVertex:%u \n", counter,
           command->count, command->firstIndex, command->baseVertex);
    // printf("counter: %d handle: %lu slice: %f\n",
    //       counter, texaddr->m_container_handle, texaddr->m_tex_page);
  }

  printf("objectcount: %u/%ld \n", objectCount, commandsBuffer.size());

  int mapRange = objectCount;

  mapRange = std::min(128, ((mapRange / 4) + 1) * 4);
  // m_TransformBuffer.bindBufferRange(0, maprange);
  // m_TexAddressBuffer.bindBufferRange(1, maprange);

  // We didn't use MAP_COHERENT here - make sure data is on the gpu
  glAssert(gl->glMemoryBarrier(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT));

  // draw
  printf("head: %ld headoffset %p objectcount: %u\n", commandsBuffer.getHead(),
         commandsBuffer.headOffset(), objectCount);

  indexBuffer.bind();
  glAssert(gl->glMultiDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT,
                                           commandsBuffer.headOffset(),
                                           objectCount, 0));
  indexBuffer.unbind();

  commandsBuffer.onUsageComplete(mapRange);
  // transformBuffer.onUsageComplete(maprange);
  // texAddressBuffer.onUsageComplete(maprange);

  glAssert(gl->glBindVertexArray(0));
}

}  // namespace Graphics
