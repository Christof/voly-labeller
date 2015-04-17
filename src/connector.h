#ifndef SRC_CONNECTOR_H_

#define SRC_CONNECTOR_H_

#include <Eigen/Core>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLBuffer>
#include <memory>
#include <string>
#include <vector>
#include "./gl.h"
#include "./render_data.h"
#include "./shader_program.h"

/**
 * \brief
 *
 *
 */
class Connector
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Connector(Eigen::Vector3f anchor, Eigen::Vector3f label);
  virtual ~Connector();

  void render(Gl *gl, const RenderData &renderData);

  void initialize(Gl *gl);

  Eigen::Vector4f color;

 private:
  Eigen::Vector3f anchor;
  Eigen::Vector3f label;

  QOpenGLVertexArrayObject vertexArrayObject;
  std::vector<QOpenGLBuffer> buffers;
  std::shared_ptr<ShaderProgram> shaderProgram;

  template <class ElementType>
  void createBuffer(QOpenGLBuffer::Type bufferType, ElementType *data,
                    std::string usage, int perVertexElements,
                    int numberOfVertices);
};

#endif  // SRC_CONNECTOR_H_
