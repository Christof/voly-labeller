#ifndef SRC_TEXTURE_MAPPER_MANAGER_CONTROLLER_H_

#define SRC_TEXTURE_MAPPER_MANAGER_CONTROLLER_H_

#include <QObject>
#include <memory>
#include "./texture_mapper_manager.h"

/**
 * \brief Provides an interface to the TextureMapperManager for the UI
 *
 *
 */
class TextureMapperManagerController : public QObject
{
  Q_OBJECT
 public:
  TextureMapperManagerController(
      std::shared_ptr<TextureMapperManager> textureMapperManager);

 public slots:
  void saveDistanceTransform();
  void saveApollonius();
  void saveSaliency();

 private:
  std::shared_ptr<TextureMapperManager> textureMapperManager;
};

#endif  // SRC_TEXTURE_MAPPER_MANAGER_CONTROLLER_H_
