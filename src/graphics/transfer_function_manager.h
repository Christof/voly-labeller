#ifndef SRC_GRAPHICS_TRANSFER_FUNCTION_MANAGER_H_

#define SRC_GRAPHICS_TRANSFER_FUNCTION_MANAGER_H_

#include <string>
#include <memory>
#include <map>
#include "./texture_address.h"

namespace Graphics
{

class TextureManager;

/**
 * \brief Provides access to a large lookup table wherein each row represents a
 * transfer function
 *
 * New transfer functions can be added from a gradient file (.gra) using #add().
 */
class TransferFunctionManager
{
 public:
  TransferFunctionManager(std::shared_ptr<TextureManager> textureManager);
  virtual ~TransferFunctionManager();

  /**
   * \brief Generates a transfer function lookup row from the given gradient
   * file
   *
   * \return The row in the transfer function lookup table.
   */
  int add(std::string path);

  TextureAddress getTextureAddress();

 private:
  const int width = 4096;
  const int height = 64;
  std::shared_ptr<TextureManager> textureManager;
  // <path, row>
  std::map<std::string, int> rowsCache;
  int textureId = -1;
  int usedRows = 0;
};

}  // namespace Graphics

#endif  // SRC_GRAPHICS_TRANSFER_FUNCTION_MANAGER_H_
