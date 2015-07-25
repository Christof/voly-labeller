#include "./buffer_manager.h"
#include <QLoggingCategory>
#include <vector>
#include <algorithm>
#include "./gl.h"
#include "./texture_manager.h"

namespace Graphics
{

QLoggingCategory bmChan("Graphics.BufferManager");

BufferManager::BufferManager()
  : positionBuffer(3, sizeof(float), GL_FLOAT),
    normalBuffer(3, sizeof(float), GL_FLOAT),
    colorBuffer(4, sizeof(float), GL_FLOAT),
    texCoordBuffer(2, sizeof(float), GL_FLOAT),
    drawIdBuffer(1, sizeof(uint), GL_UNSIGNED_INT),
    indexBuffer(1, sizeof(uint), GL_UNSIGNED_INT), vertexBufferManager(0),
    indexBufferManager(0)
{
}

BufferManager::~BufferManager()
{
  if (!gl)
    return;

  glAssert(gl->glDeleteVertexArrays(1, &vertexArrayId));
}

void BufferManager::initialize(Gl *gl, uint maxObjectCount, uint bufferSize)
{
  this->gl = gl;

  vertexBufferManager = BufferHoleManager(bufferSize);
  indexBufferManager = BufferHoleManager(bufferSize);

  glAssert(gl->glGenVertexArrays(1, &vertexArrayId));
  glAssert(gl->glBindVertexArray(vertexArrayId));

  positionBuffer.initialize(gl, bufferSize);
  normalBuffer.initialize(gl, bufferSize);
  colorBuffer.initialize(gl, bufferSize);
  texCoordBuffer.initialize(gl, bufferSize);

  indexBuffer.initialize(gl, bufferSize, GL_ELEMENT_ARRAY_BUFFER);

  initializeDrawIdBuffer(maxObjectCount);

  // bind per vertex attibutes to vertex array
  positionBuffer.bindAttrib(0);
  normalBuffer.bindAttrib(1);
  colorBuffer.bindAttrib(2);
  texCoordBuffer.bindAttrib(3);
  drawIdBuffer.bindAttribDivisor(4, 1);

  glAssert(gl->glBindVertexArray(0));
}

void BufferManager::initializeDrawIdBuffer(uint maxObjectCount)
{
  drawIdBuffer.initialize(gl, maxObjectCount);
  std::vector<uint> drawids;
  uint idval = 0;
  drawids.resize(maxObjectCount);

  for_each(drawids.begin(), drawids.end(),
           [&idval](std::vector<uint>::value_type &v)
           {
    v = idval++;
  });

  drawIdBuffer.setData(drawids);
}

BufferInformation BufferManager::addObject(const std::vector<float> &vertices,
                                           const std::vector<float> &normals,
                                           const std::vector<float> &colors,
                                           const std::vector<float> &texCoords,
                                           const std::vector<uint> &indices)
{
  qCDebug(bmChan) << "add object";
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

  BufferInformation bufferInformation;
  bool reserve_success = vertexBufferManager.reserve(
      vertexCount, bufferInformation.vertexBufferOffset);
  if (reserve_success)
  {
    reserve_success = indexBufferManager.reserve(
        indices.size(), bufferInformation.indexBufferOffset);
    if (!reserve_success)
    {
      vertexBufferManager.release(bufferInformation.vertexBufferOffset);
    }
  }

  if (!reserve_success)
    throw std::runtime_error("Failed to reserve space in buffers");

  // fill buffers
  positionBuffer.setData(vertices, bufferInformation.vertexBufferOffset);
  normalBuffer.setData(normals, bufferInformation.vertexBufferOffset);
  colorBuffer.setData(colors, bufferInformation.vertexBufferOffset);
  texCoordBuffer.setData(texCoords, bufferInformation.vertexBufferOffset);

  indexBuffer.setData(indices, bufferInformation.indexBufferOffset);

  return bufferInformation;
}

void BufferManager::bind()
{
  glAssert(gl->glBindVertexArray(vertexArrayId));
  indexBuffer.bind();
}

void BufferManager::unbind()
{
  indexBuffer.unbind();
  glAssert(gl->glBindVertexArray(0));
}

}  // namespace Graphics
