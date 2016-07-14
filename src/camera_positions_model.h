#ifndef SRC_CAMERA_POSITIONS_MODEL_H_

#define SRC_CAMERA_POSITIONS_MODEL_H_

#include <QAbstractListModel>
#include <memory>
#include "./camera_node.h"

class Nodes;

/**
 * \brief Model to add, rename, remove and move to camera positions
 *
 */
class CameraPositionsModel : public QAbstractTableModel
{
  Q_OBJECT
 public:
  explicit CameraPositionsModel(std::shared_ptr<Nodes> nodes);
  // virtual ~CameraPositionsModel();

  enum CameraPositionRoles
  {
    NameRole = Qt::UserRole + 1,
  };

  QHash<int, QByteArray> roleNames() const Q_DECL_OVERRIDE;

  int rowCount(const QModelIndex &parent = QModelIndex()) const Q_DECL_OVERRIDE;
  int
  columnCount(const QModelIndex &parent = QModelIndex()) const Q_DECL_OVERRIDE;

  QVariant data(const QModelIndex &index,
                int role = Qt::DisplayRole) const Q_DECL_OVERRIDE;

  Qt::ItemFlags flags(const QModelIndex &index) const Q_DECL_OVERRIDE;

  QVariant headerData(int section, Qt::Orientation orientation,
                      int role = Qt::DisplayRole) const Q_DECL_OVERRIDE;

  void update(std::vector<CameraPosition> cameraPositions);

  Q_PROPERTY(bool isVisible MEMBER isVisible READ getIsVisible NOTIFY
                 isVisibleChanged)

  bool getIsVisible() const;

 public slots:
  void save();
  void changeName(int row, QString text);
  void deletePosition(int row);
  void toggleVisibility();
  void moveTo(int row);
  void setTo(int row);

 signals:
  void isVisibleChanged();

 private:
  std::shared_ptr<Nodes> nodes;
  std::vector<CameraPosition> cameraPositions;
  bool isVisible = false;
  volatile bool ignoreNextLabelUpdate = false;
};

#endif  // SRC_CAMERA_POSITIONS_MODEL_H_
