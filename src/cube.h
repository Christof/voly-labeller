#ifndef SRC_CUBE_H_

#define SRC_CUBE_H_

#include <Eigen/Core>
#include <vector>
#include "./render_data.h"
#include "./renderable.h"

/**
 * \brief
 *
 *
 */
class Cube : public Renderable
{
 public:
  Cube();

 protected:
  virtual void createBuffers(std::shared_ptr<RenderObject> renderObject);
  virtual void setUniforms(std::shared_ptr<ShaderProgram> shaderProgram,
                           const RenderData &renderData);
  virtual void draw(Gl *gl);

 private:
  std::vector<Eigen::Vector3f> points;
  static const int indexCount = 36;
};

#endif  // SRC_CUBE_H_
