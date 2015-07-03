#ifndef SRC_GRAPHICS_HA_BUFFER_H_

#define SRC_GRAPHICS_HA_BUFFER_H_

#include <Eigen/Core>
#include <memory>
#include "./render_data.h"
#include "./gl.h"
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
#define TIMING_THRESHOLD 20.0f
#define USE_TEXTURE 1

class ShaderProgram;

/**
 * \brief
 *
 *
 */
class HABuffer
{
 public:
  explicit HABuffer(Eigen::Vector2i size);

  void initialize(Gl *gl);

  // build is now begin() and end(); because in the middle the models are
  // rendered
  void begin(const RenderData &renderData);
  bool end();
  void render();
  void clear();

  /*
  void setOrtho(Eigen::Matrix4f &mat, float l, float r, float b, float t,
                float zn, float zf);
  void setPerspective(Eigen::Matrix4f &mat, float fov, float aspect,
                      float znear, float zfar);
                      */

 private:
  void initializeShadersHash();
  void initializeBufferHash();
  void displayStatistics(const char *label);

  Eigen::Vector2i size;
  Gl *gl;
  std::unique_ptr<ShaderProgram> buildShader;
  std::unique_ptr<ShaderProgram> renderShader;
  std::unique_ptr<ShaderProgram> clearShader;

  unsigned int habufferScreenSize = 0;
  unsigned int habufferTableSize = 0;
  uint habufferNumRecords = 0;
  uint habufferCountsSize = 0;
  uint habufferLoopCount = 0;

  Buffer RecordsBuffer;
  Buffer CountsBuffer;
  Buffer FragmentDataBuffer;

  float habufferZNear = 0.01f;
  float habufferZFar = 20.0f;
  float habufferOpacity = 0.5f;
  float habufferLightPos[3] = { 0.0f, 0.0f, 0.0f };
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_HA_BUFFER_H_
