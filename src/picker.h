#ifndef SRC_PICKER_H_

#define SRC_PICKER_H_

#include <Eigen/Core>
#include <memory>
#include "./graphics/frame_buffer_object.h"
#include "./graphics/gl.h"

class Labels;

/**
 * \brief Performs picking in the depth buffer and updates label anchor position
 *
 * To trigger the picking #pick must be called. Every frame #doPick must be
 * invoked, which performs the picking if #pick was called previously.
 *
 * After instantiation and on every size change #resize must be called.
 */
class Picker
{
 public:
  Picker(std::shared_ptr<Graphics::FrameBufferObject> fbo, Graphics::Gl *gl,
         std::shared_ptr<Labels> labels);

  void pick(int id, Eigen::Vector2f position);
  void doPick(Eigen::Matrix4f viewProjection);
  void resize(int width, int height);

 private:
  std::shared_ptr<Graphics::FrameBufferObject> fbo;
  Graphics::Gl *gl;
  std::shared_ptr<Labels> labels;

  Eigen::Vector2f pickingPosition;
  int pickingLabelId;
  bool performPicking;

  int width;
  int height;
};

#endif  // SRC_PICKER_H_
