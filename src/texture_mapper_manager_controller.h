#ifndef SRC_TEXTURE_MAPPER_MANAGER_CONTROLLER_H_

#define SRC_TEXTURE_MAPPER_MANAGER_CONTROLLER_H_

#include <memory>
#include <QObject>
#include "./texture_mapper_manager.h"

/**
 * \brief
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
  void saveOccupancy();
  void savePositions();
  void saveDistanceTransform();
  void saveApollonius();

 private:
  std::shared_ptr<TextureMapperManager> textureMapperManager;
};

#endif  // SRC_TEXTURE_MAPPER_MANAGER_CONTROLLER_H_
