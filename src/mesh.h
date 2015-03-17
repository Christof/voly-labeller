#ifndef SRC_MESH_H_

#define SRC_MESH_H_

#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLBuffer>
#include <vector>
#include <assimp/scene.h>
#include <assimp/material.h>
#include <Eigen/Core>
#include "./shader_program.h"

class QOpenGLFunctions_4_3_Core;
/**
 * \brief Encapsulates a single mesh including its material.
 *
 */
class Mesh
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Mesh(QOpenGLFunctions_4_3_Core *gl, aiMesh *mesh, aiMaterial *material);
  virtual ~Mesh();

  void render(Eigen::Matrix4f projection, Eigen::Matrix4f view);

 private:
  QOpenGLFunctions_4_3_Core *gl;
  QOpenGLVertexArrayObject vertexArrayObject;
  std::vector<QOpenGLBuffer> buffers;
  ShaderProgram shaderProgram;

  template <class ElementType>
  void createBuffer(QOpenGLBuffer::Type bufferType, ElementType *data,
                    std::string usage, int perVertexElements,
                    int numberOfVertices);

  Eigen::Vector4f loadVector4FromMaterial(const char *key,
                                          aiMaterial *material);
  float loadFloatFromMaterial(const char *key, aiMaterial *material);

  int vertexCount;
  int indexCount;
  Eigen::Vector4f ambientColor;
  Eigen::Vector4f diffuseColor;
  Eigen::Vector4f specularColor;
  float shininess;
};

#endif  // SRC_MESH_H_
