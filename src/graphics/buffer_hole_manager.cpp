#include "./buffer_hole_manager.h"
#include <QLoggingCategory>

namespace Graphics
{

QLoggingCategory bhmChan("Graphics.BufferHoleManager");

BufferHoleManager::BufferHoleManager(int bufferSize) : bufferSize(bufferSize)
{
  holes[0] = bufferSize;
}

bool BufferHoleManager::reserve(uint requestSize, uint &offset)
{
  if (requestSize > bufferSize)
    return false;

  offset = -1;

  // FIXME: try to find optimal hole to avoid fragmentation
  for (auto &hole : holes)
  {
    const uint holeStart = hole.first;
    const uint holeSize = hole.second;

    if (holeSize >= requestSize)
    {
      reservedChunks[holeStart] = requestSize;

      if (holeSize > requestSize)
        holes[holeStart + requestSize] = holeSize - requestSize;

      // remove old hole
      holes.erase(holeStart);

      offset = holeStart;

      return true;
    }
  }

  return false;
}

bool BufferHoleManager::release(uint offset)
{
  auto reserved = reservedChunks.find(offset);

  if (reserved == reservedChunks.end())
    return false;

  const uint chunkSize = reserved->second;
  reservedChunks.erase(offset);
  holes[offset] = chunkSize;

  tryJoiningWithNextHole(offset, chunkSize);
  tryJoiningWithPreviousHole(offset);

  return true;
}

void BufferHoleManager::tryJoiningWithNextHole(uint offset, uint chunkSize)
{
  auto hole = holes.find(offset + chunkSize);
  if (hole != holes.end())
  {
    qCDebug(bhmChan) << "joining next hole at" << hole->first << "("
                     << hole->second
                     << "). New size:" << hole->second + chunkSize;

    const uint nextHoleSize = hole->second;
    holes.erase(hole->first);
    holes[offset] = nextHoleSize + chunkSize;
  }
}

void BufferHoleManager::tryJoiningWithPreviousHole(uint offset)
{
  auto hole = --holes.find(offset);

  if ((hole->first + hole->second) == offset)
  {
    qCDebug(bhmChan) << "joining previous hole at" << hole->first << "("
                     << hole->second
                     << "). New size:" << hole->second + holes[offset];

    hole->second += holes[offset];
    holes.erase(offset);
  }
}
}  // namespace Graphics
