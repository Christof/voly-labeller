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
    EnabledRole
  };

  QHash<int, QByteArray> roleNames() const Q_DECL_OVERRIDE;

  int rowCount(const QModelIndex &parent = QModelIndex()) const Q_DECL_OVERRIDE;
  int
  columnCount(const QModelIndex &parent = QModelIndex()) const Q_DECL_OVERRIDE;

  QVariant data(const QModelIndex &index,
                int role = Qt::DisplayRole) const Q_DECL_OVERRIDE;

  Q_INVOKABLE bool setData(const QModelIndex &index, const QVariant &value,
                           int role) Q_DECL_OVERRIDE;

  Qt::ItemFlags flags(const QModelIndex &index) const Q_DECL_OVERRIDE;

  QVariant headerData(int section, Qt::Orientation orientation,
                      int role = Qt::DisplayRole) const Q_DECL_OVERRIDE;
 public slots:
  void changeEnabled(int row, QVariant newValue);
  void changeWeight(int row, QVariant newValue);

 private:
  std::shared_ptr<Forces::Labeller> labeller;
};

#endif  // SRC_LABELLER_CONTEXT_H_
