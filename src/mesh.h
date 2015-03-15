#ifndef SRC_MESH_H_

#define SRC_MESH_H_

#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLBuffer>
#include <vector>
#include <assimp/scene.h>
#include <Eigen/Core>

class QOpenGLFunctions_4_3_Core;
/**
 * \brief 
 *
 * 
 */
class Mesh
{
public:
  Mesh(QOpenGLFunctions_4_3_Core *gl, aiMesh *mesh);
  virtual ~Mesh();

  void render(Eigen::Matrix4f projection, Eigen::Matrix4f view);

private:
  QOpenGLFunctions_4_3_Core *gl;
  QOpenGLShaderProgram shaderProgram;
  QOpenGLVertexArrayObject vertexArrayObject;
  std::vector<QOpenGLBuffer> buffers;

  int numVerts;

  void prepareShaderProgram();
};

#endif  // SRC_MESH_H_
