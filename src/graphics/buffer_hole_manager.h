#ifndef SRC_GRAPHICS_BUFFER_HOLE_MANAGER_H_

#define SRC_GRAPHICS_BUFFER_HOLE_MANAGER_H_

#include <map>

namespace Graphics
{

/**
 * \brief Manages holes in a buffer to get better utilization
 *
 */
class BufferHoleManager
{
 public:
  BufferHoleManager(int bufferSize);
  bool reserve(uint requestsize, uint &offset);
  bool release(uint offset);

 private:
  uint bufferSize;
  std::map<uint, uint> holes;
  std::map<uint, uint> reservedChunks;

  void tryJoiningWithNextHole(uint offset, uint chunkSize);
  void tryJoiningWithPreviousHole(uint offset);
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_BUFFER_HOLE_MANAGER_H_
