#if _WIN32
#pragma warning(disable : 4522)
#endif

#include "./picker.h"
#include <Eigen/Geometry>
#include "./eigen_qdebug.h"
#include "./labelling/labels.h"
#include "./src/math/eigen.h"

Picker::Picker(std::shared_ptr<Graphics::FrameBufferObject> fbo,
               Graphics::Gl *gl, std::shared_ptr<Labels> labels)
  : fbo(fbo), gl(gl), labels(labels)
{
}

void Picker::pick(int id, Eigen::Vector2f position)
{
  pickingPosition = position;
  performPicking = true;
  pickingLabelId = id;
}

void Picker::doPick(Eigen::Matrix4f viewProjection)
{
  if (!performPicking)
    return;

  float depth = -2.0f;

  fbo->bindDepthTexture(GL_TEXTURE0);

  gl->glReadPixels(pickingPosition.x(), height - pickingPosition.y() - 1, 1, 1,
                   GL_DEPTH_COMPONENT, GL_FLOAT, &depth);
  Eigen::Vector4f positionNDC(pickingPosition.x() * 2.0f / width - 1.0f,
                              pickingPosition.y() * -2.0f / height + 1.0f,
                              depth * 2.0f - 1.0f, 1.0f);

  Eigen::Vector4f positionWorld = viewProjection.inverse() * positionNDC;
  positionWorld = positionWorld / positionWorld.w();

  qWarning() << "picked:" << positionWorld << "depth value:" << depth;

  performPicking = false;
  auto label = labels->getById(pickingLabelId);
  label.anchorPosition = toVector3f(positionWorld);

  labels->update(label);
}

void Picker::resize(int width, int height)
{
  this->width = width;
  this->height = height;
}

