#ifndef SRC_LABELS_MODEL_H_

#define SRC_LABELS_MODEL_H_

#include <QAbstractTableModel>
#include <memory>

class Nodes;

/**
 * \brief
 *
 *
 */
class LabelsModel : public QAbstractTableModel
{
 public:
  explicit LabelsModel(std::shared_ptr<Nodes> nodes);

  enum ForceRoles
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

  Q_PROPERTY(bool isVisible MEMBER isVisible READ getIsVisible NOTIFY
                 isVisibleChanged)

  bool getIsVisible() const;

 private:
  std::shared_ptr<Nodes> nodes;
  bool isVisible = true;
};

#endif  // SRC_LABELS_MODEL_H_
