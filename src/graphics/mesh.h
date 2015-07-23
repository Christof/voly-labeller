#ifndef SRC_GRAPHICS_MESH_H_

#define SRC_GRAPHICS_MESH_H_

#include <assimp/scene.h>
#include <assimp/material.h>
#include <Eigen/Core>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLBuffer>
#include <vector>
#include <memory>
#include <string>
#include "./render_data.h"
#include "../math/obb.h"
#include "./renderable.h"

namespace Graphics
{

class Gl;

/**
 * \brief Encapsulates a single mesh including its material.
 *
 */
class Mesh : public Renderable
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Mesh(aiMesh *mesh, aiMaterial *material);
  virtual ~Mesh();

  std::shared_ptr<Math::Obb> obb;

 protected:
  virtual void createBuffers(std::shared_ptr<RenderObject> renderObject);
  virtual void setUniforms(std::shared_ptr<ShaderProgram> shaderProgram,
                           const RenderData &renderData);
  virtual void draw(Gl *gl);

 private:
  void createObb();
  Eigen::Vector4f loadVector4FromMaterial(const char *key,
                                          aiMaterial *material);
  float loadFloatFromMaterial(const char *key, aiMaterial *material);

  int vertexCount;
  int indexCount;
  unsigned int *indexData;
  float *positionData;
  float *normalData;
  float *textureCoordinateData;
  Eigen::Vector4f ambientColor;
  Eigen::Vector4f diffuseColor;
  Eigen::Vector4f specularColor;
  float shininess;

  int id;
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_MESH_H_
