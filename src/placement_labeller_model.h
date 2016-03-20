#ifndef SRC_PLACEMENT_LABELLER_MODEL_H_

#define SRC_PLACEMENT_LABELLER_MODEL_H_

#include <QAbstractTableModel>
#include <memory>
#include "./placement/cost_function_calculator.h"

class LabellingCoordinator;

/**
 * \brief Model to change weights and settings of placement labeller
 *
 */
class PlacementLabellerModel : public QAbstractTableModel
{
  Q_OBJECT
 public:
  explicit PlacementLabellerModel(
      std::shared_ptr<LabellingCoordinator> coordinator);

  enum WeightRoles
  {
    NameRole = Qt::UserRole + 1,
    WeightRole,
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

 public slots:
  void changeWeight(int row, QVariant newValue);
  void toggleVisibility();

 signals:
  void isVisibleChanged();

 private:
  std::shared_ptr<LabellingCoordinator> coordinator;

  Placement::CostFunctionWeights weights;

  QString getWeightNameForRowIndex(int rowIndex) const;
  float getWeightValueForRowIndex(int rowIndex) const;

  bool isVisible = false;
};

#endif  // SRC_PLACEMENT_LABELLER_MODEL_H_
