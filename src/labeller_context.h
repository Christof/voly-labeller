#ifndef SRC_LABELLER_CONTEXT_H_

#define SRC_LABELLER_CONTEXT_H_

#include <QAbstractListModel>
#include <QList>
#include <memory>
#include "./forces/labeller.h"

/**
 * \brief
 *
 *
 */
class LabellerContext : public QAbstractListModel
{
  Q_OBJECT
 public:
  LabellerContext(std::shared_ptr<Forces::Labeller> labeller);

  enum ForceRoles
  {
    NameRole = Qt::UserRole + 1,
    EnabledRole,
    WeightRole
  };

  QHash<int, QByteArray> roleNames() const;

  int rowCount(const QModelIndex &parent = QModelIndex()) const;

  QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const;

 private:
  std::shared_ptr<Forces::Labeller> labeller;
};

#endif  // SRC_LABELLER_CONTEXT_H_
