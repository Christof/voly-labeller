#ifndef SRC_MESH_H_

#define SRC_MESH_H_

#include <assimp/scene.h>
#include <assimp/material.h>
#include <Eigen/Core>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLBuffer>
#include <vector>
#include <memory>
#include <string>
#include "./shader_program.h"
#include "./gl.h"

/**
 * \brief Encapsulates a single mesh including its material.
 *
 */
class Mesh
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Mesh(Gl *gl, aiMesh *mesh, aiMaterial *material);
  virtual ~Mesh();

  void render(Eigen::Matrix4f projection, Eigen::Matrix4f view);

 private:
  Gl *gl;
  QOpenGLVertexArrayObject vertexArrayObject;
  std::vector<QOpenGLBuffer> buffers;
  std::unique_ptr<ShaderProgram> shaderProgram;

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
