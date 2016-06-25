#include "./object_data.h"
#include <cassert>

namespace Graphics
{

ObjectData::ObjectData(int id, int vertexOffset, int indexOffset, int indexSize,
                       int shaderProgramId, int primitiveType)
  : modelMatrix(Eigen::Matrix4f::Identity()), id(id),
    primitiveType(primitiveType), vertexOffset(vertexOffset),
    indexOffset(indexOffset), indexSize(indexSize),
    shaderProgramId(shaderProgramId), customBuffers(1)

{
}

ObjectData::ObjectData() : ObjectData(-1, -1, -1, -1, -1, -1)
{
}

int ObjectData::getId() const
{
  return id;
}

int ObjectData::getPrimitiveType() const
{
  return primitiveType;
}

int ObjectData::getVertexOffset() const
{
  return vertexOffset;
}

int ObjectData::getIndexOffset() const
{
  return indexOffset;
}

int ObjectData::getIndexSize() const
{
  return indexSize;
}

int ObjectData::getShaderProgramId() const
{
  return shaderProgramId;
}

int ObjectData::getCustomBufferSize(int index) const
{
  return customBuffers[index].size;
}

bool ObjectData::hasCustomBuffer() const
{
  return customBuffers[0].setBuffer && customBuffers[0].size;
}

bool ObjectData::isInitialized()
{
  return id != -1;
}

void ObjectData::setCustomBuffer(int size,
                                 std::function<void(void *)> setFunction)
{
  setCustomBufferFor(0, size, setFunction);
}

void ObjectData::setCustomBufferFor(int index, int size,
                                    std::function<void(void *)> setFunction)
{
  customBuffers[index].size = size;
  customBuffers[index].setBuffer = setFunction;
}

void ObjectData::fillBufferElement(void *bufferStart, int index)
{
  fillBufferElementFor(0, bufferStart, index);
}

void ObjectData::fillBufferElementFor(int customBufferIndex, void *bufferStart,
                                      int index)
{
  CustomBufferData &customBuffer = customBuffers[customBufferIndex];
  assert(customBuffer.size != 0);
  customBuffer.setBuffer(static_cast<char *>(bufferStart) +
                         index * customBuffer.size);
}

}  // namespace Graphics
