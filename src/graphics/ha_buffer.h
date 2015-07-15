#ifndef SRC_GRAPHICS_HA_BUFFER_H_

#define SRC_GRAPHICS_HA_BUFFER_H_

#include <Eigen/Core>
#include <memory>
#include "./render_data.h"
#include "./gl.h"
#include "./timer.h"
#include "./buffer.h"

namespace Graphics
{

// TODO(SIRK) change with size
struct FragmentData
{
  float color[4];
  float pos[4];
};

#define USE_INDIRECT 1
#define TIMING_THRESHOLD 1.0f
#define USE_TEXTURE 1

class ShaderProgram;
class Quad;

/**
 * \brief
 *
 *
 */
class HABuffer
{
 public:
  explicit HABuffer(Eigen::Vector2i size);
  ~HABuffer();

  void initialize(Gl *gl);
  void updateNearAndFarPlanes(float near, float far);

  void beginAll();
  // build is now begin() and end(); because in the middle the models are
  // rendered
  void begin(std::shared_ptr<ShaderProgram> shader);
  bool endAll();
  void render();
  void clear();

 private:
  void initializeShadersHash();
  void initializeBufferHash();
  void setUniforms(std::shared_ptr<ShaderProgram> shader);
  void displayStatistics(const char *label);

  Eigen::Vector2i size;
  Gl *gl;
  std::shared_ptr<Quad> quad;
  std::shared_ptr<ShaderProgram> renderShader;
  std::shared_ptr<ShaderProgram> clearShader;

  unsigned int habufferScreenSize = 0;
  unsigned int habufferTableSize = 0;
  uint habufferNumRecords = 0;
  uint habufferCountsSize = 0;
  uint habufferLoopCount = 0;

  Buffer RecordsBuffer;
  Buffer CountsBuffer;
  Buffer FragmentDataBuffer;

  float zNear = 0.1f;
  float zFar = 5.0f;
  float habufferOpacity = 0.5f;
  float habufferLightPos[3] = { 0.0f, 0.0f, 0.0f };
  uint *offsets;

  int lastUsedProgram = 0;

  Timer clearTimer;
  Timer buildTimer;
  Timer renderTimer;
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_HA_BUFFER_H_
