#ifndef SRC_LABELLING_CONTROLLER_H_

#define SRC_LABELLING_CONTROLLER_H_

#include <QObject>
#include <memory>

class LabellingCoordinator;

/**
 * \brief Provides facade for UI to change labelling settings
 *
 */
class LabellingController : public QObject
{
  Q_OBJECT
 public:
  LabellingController(
      std::shared_ptr<LabellingCoordinator> labellingCoordinator);

 public slots:
  void toggleInternalLabelling();
  void toggleAnchorVisibility();
  void toggleForces();
  void toggleOptimizeOnIdle();
  void toggleApollonius();
  void saveOcclusion();
  void saveConstraints();

 private:
  std::shared_ptr<LabellingCoordinator> labellingCoordinator;
};

#endif  // SRC_LABELLING_CONTROLLER_H_
