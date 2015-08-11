#ifndef SRC_GRAPHICS_OBJECT_DATA_H_

#define SRC_GRAPHICS_OBJECT_DATA_H_

#include <Eigen/Core>
#include <functional>

namespace Graphics
{

struct ObjectData
{
  ObjectData()
    : primitiveType(-1), vertexOffset(-1), indexOffset(-1), indexSize(-1),
      shaderProgramId(-1), customBufferSize(0), setBuffer(nullptr),
      transform(Eigen::Matrix4f::Identity())
  {
  }

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
    return primitiveType != -1;
  }
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_OBJECT_DATA_H_
