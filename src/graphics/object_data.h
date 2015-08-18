#ifndef SRC_GRAPHICS_OBJECT_DATA_H_

#define SRC_GRAPHICS_OBJECT_DATA_H_

#include <Eigen/Core>
#include <functional>

namespace Graphics
{

struct ObjectData
{
  ObjectData(int id, int vertexOffset, int indexOffset, int indexSize,
             int shaderProgramId, int primitiveType)
    : id(id), primitiveType(primitiveType), vertexOffset(vertexOffset),
      indexOffset(indexOffset), indexSize(indexSize),
      shaderProgramId(shaderProgramId), customBufferSize(0), setBuffer(nullptr),
      transform(Eigen::Matrix4f::Identity())
  {
  }

  ObjectData()
    : id(-1), primitiveType(-1), vertexOffset(-1), indexOffset(-1), indexSize(-1),
      shaderProgramId(-1), customBufferSize(0), setBuffer(nullptr),
      transform(Eigen::Matrix4f::Identity())
  {
  }

  int id;
  int primitiveType;
  int vertexOffset;
  int indexOffset;
  int indexSize;

  int shaderProgramId;

  int customBufferSize;
  std::function<void(void *)> setBuffer;

  Eigen::Matrix4f transform;

  bool isInitialized()
  {
    return id != -1;
  }
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_OBJECT_DATA_H_
