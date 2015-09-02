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
#include "./object_manager.h"

namespace Graphics
{

class Gl;
class ObjectManager;

/**
 * \brief Description of phong material consisting of ambient, diffuse and specular
 * color as well as a shininess value
 */
struct PhongMaterial
{
  Eigen::Vector4f ambientColor;
  Eigen::Vector4f diffuseColor;
  Eigen::Vector4f specularColor;
  float shininess;
};

/**
 * \brief Encapsulates a single mesh including its material
 */
class Mesh : public Renderable
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Mesh(aiMesh *mesh, aiMaterial *material);
  virtual ~Mesh();

  std::shared_ptr<Math::Obb> obb;

 protected:
  virtual ObjectData
  createBuffers(std::shared_ptr<ObjectManager> objectManager,
                std::shared_ptr<TextureManager> textureManager,
                std::shared_ptr<ShaderManager> shaderManager);

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
  PhongMaterial phongMaterial;

  bool hasTexture;
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_MESH_H_
