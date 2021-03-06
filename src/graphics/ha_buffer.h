#ifndef SRC_GRAPHICS_HA_BUFFER_H_

#define SRC_GRAPHICS_HA_BUFFER_H_

#include <Eigen/Core>
#include <memory>
#include <vector>
#include "./render_data.h"
#include "./gl.h"
#include "./timer.h"
#include "./buffer.h"

namespace Graphics
{

class ShaderProgram;
class Managers;
class ScreenQuad;

/**
 * \brief Coherent hashing A-Buffer to render transparent objects without
 * sorting
 *
 * Implemented after:
 * Lefebvre, Sylvain, Samuel Hornus, and Anass Lasram. "Per-Pixel Lists for
 * Single Pass A-Buffer." GPU Pro 5: Advanced Rendering Techniques (2014).
 *
 * The Buffer must be initialized once by calling HABuffer::initialize. Each
 * frame the buffer must be cleared by means of HABuffer::clearAndPrepare. For
 * each object which is rendered HABuffer::begin must by called with the shader
 * of the object. After all objects have been rendered into the HABuffer, the
 * buffer itself can be resolved and rendered by calling HABuffer::render.
 *
 * Fragment shader for object which are rendered into the HABuffer have to
 * implement a certain interface. The following skeleton describes the
 *requirements:
 *
 * \code{glsl}
 *#version 440
 *#include "HABufferImplementation.hglsl"
 *
 *
 *FragmentData computeData()
 *{
 *  ...
 *
 *  FragmentData data;
 *  data.color = color;
 *  data.eyePos = outEyePosition;
 *
 *  return data;
 *}
 *
 *#include "buildHABuffer.hglsl"
 *\endcode
 *
 * So the two hglsl files must be included and the `computeData` function
 * must be implemented which sets the color and position.
 */
class HABuffer
{
 public:
  explicit HABuffer(Eigen::Vector2i size);
  ~HABuffer();

  void initialize(Gl *gl, std::shared_ptr<Managers> managers);
  void updateNearAndFarPlanes(float near, float far);

  void clearAndPrepare(std::shared_ptr<Managers> managers);
  void begin(std::shared_ptr<ShaderProgram> shader);
  void render(std::shared_ptr<Graphics::Managers> managers,
              const RenderData &renderData);

  void setLayerZValues(std::vector<float> layerZValues, int layerCount);

  bool wireframe = false;

 private:
  void initializeBufferHash();
  void setUniforms(std::shared_ptr<ShaderProgram> shader);
  void syncAndGetCounts();
  void displayStatistics(const char *label);
  void setLayeringUniforms(std::shared_ptr<ShaderProgram> renderShader,
                           const RenderData &renderData);

  // vec4 color and vec4 position
  const int FRAGMENT_DATA_SIZE = 36;

  Eigen::Vector2i size;
  Gl *gl;
  std::shared_ptr<ScreenQuad> clearQuad;
  std::shared_ptr<ScreenQuad> renderQuad;

  unsigned int habufferScreenSize = 0;
  unsigned int habufferTableSize = 0;
  uint tableElementCount = 0;
  uint habufferCountsSize = 0;

  Buffer recordsBuffer;
  Buffer countsBuffer;
  Buffer fragmentDataBuffer;

  float zNear = 0.1f;
  float zFar = 5.0f;
  uint *offsets;

  int noiseTexture;

  int lastUsedProgram = 0;

  Timer clearTimer;
  Timer buildTimer;
  Timer renderTimer;

  std::vector<float> layerZValues;
  int layerCount;
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_HA_BUFFER_H_
