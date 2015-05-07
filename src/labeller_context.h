#ifndef SRC_LABELLER_CONTEXT_H_

#define SRC_LABELLER_CONTEXT_H_

#include <QAbstractTableModel>
#include <QList>
#include <memory>
#include "./forces/labeller.h"

/**
 * \brief
 *
 *
 */
class LabellerContext : public QAbstractTableModel
{
  Q_OBJECT
 public:
  LabellerContext(std::shared_ptr<Forces::Labeller> labeller);

  enum ForceRoles
  {
    NameRole = Qt::UserRole + 1,
    WeightRole,
    EnabledRole = Qt::CheckStateRole
  };

  QHash<int, QByteArray> roleNames() const;

  int rowCount(const QModelIndex &parent = QModelIndex()) const;
  int columnCount(const QModelIndex &parent = QModelIndex()) const;

  QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const;

  bool setData(const QModelIndex &index, const QVariant &value, int role);

  Qt::ItemFlags flags(const QModelIndex &index) const;

  QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const;
 private:
  std::shared_ptr<Forces::Labeller> labeller;
};

#endif  // SRC_LABELLER_CONTEXT_H_
