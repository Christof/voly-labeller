#ifndef SRC_CAMERA_POSITIONS_MODEL_H_

#define SRC_CAMERA_POSITIONS_MODEL_H_

#include <QAbstractListModel>
#include <memory>

class Nodes;

/**
 * \brief
 *
 *
 */
class CameraPositionsModel : public QAbstractTableModel
{
  Q_OBJECT
 public:
  CameraPositionsModel(std::shared_ptr<Nodes> nodes);
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

 public slots:
  void save();
  void changeName(int row, QString text);

 private:
  std::shared_ptr<Nodes> nodes;
};

#endif  // SRC_CAMERA_POSITIONS_MODEL_H_