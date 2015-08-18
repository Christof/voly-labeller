#include "./object_data.h"
#include <cassert>

namespace Graphics
{

ObjectData::ObjectData(int id, int vertexOffset, int indexOffset, int indexSize,
                       int shaderProgramId, int primitiveType)
  : modelMatrix(Eigen::Matrix4f::Identity()), id(id),
    primitiveType(primitiveType), vertexOffset(vertexOffset),
    indexOffset(indexOffset), indexSize(indexSize),
    shaderProgramId(shaderProgramId), customBufferSize(0), setBuffer(nullptr)

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

int ObjectData::getCustomBufferSize() const
{
  return customBufferSize;
}

bool ObjectData::hasCustomBuffer() const
{
  return setBuffer && customBufferSize;
}

bool ObjectData::isInitialized()
{
  return id != -1;
}

void ObjectData::setCustomBuffer(int size,
                                 std::function<void(void *)> setFunction)
{
  customBufferSize = size;
  setBuffer = setFunction;
}

void ObjectData::fillBufferElement(void *bufferStart, int index)
{
  assert(customBufferSize != 0);
  setBuffer(static_cast<char *>(bufferStart) + index * customBufferSize);
}

}  // namespace Graphics
