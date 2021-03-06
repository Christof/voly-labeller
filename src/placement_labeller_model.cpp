#if _WIN32
#pragma warning(disable : 4996 4267)
#endif

#include "./placement_labeller_model.h"
#include <limits>
#include "./labelling_coordinator.h"

const int ROW_COUNT = 9;

PlacementLabellerModel::PlacementLabellerModel(
    std::shared_ptr<LabellingCoordinator> coordinator)
  : coordinator(coordinator)
{
}

QHash<int, QByteArray> PlacementLabellerModel::roleNames() const
{
  QHash<int, QByteArray> roles;
  roles[NameRole] = "name";
  roles[WeightRole] = "weight";

  return roles;
}

int PlacementLabellerModel::rowCount(const QModelIndex &parent) const
{
  Q_UNUSED(parent);
  return ROW_COUNT;
}

int PlacementLabellerModel::columnCount(const QModelIndex &parent) const
{
  Q_UNUSED(parent);
  return 2;
}

QVariant PlacementLabellerModel::data(const QModelIndex &index, int role) const
{
  if (index.row() < 0 || index.row() >= ROW_COUNT)
    return QVariant();

  switch (role)
  {
  case NameRole:
    return QVariant::fromValue(getWeightNameForRowIndex(index.row()));
  case WeightRole:
    return QVariant::fromValue(getWeightValueForRowIndex(index.row()));
  }
  return QVariant();
}

Qt::ItemFlags PlacementLabellerModel::flags(const QModelIndex &index) const
{
  if (!index.isValid())
    return Qt::ItemIsEnabled;

  return QAbstractItemModel::flags(index) | Qt::ItemFlag::ItemIsEditable |
         Qt::ItemIsUserCheckable;
}

bool PlacementLabellerModel::getIsVisible() const
{
  return isVisible;
}

void PlacementLabellerModel::changeWeight(int row, QVariant newValue)
{
  bool converted = false;
  auto value = newValue.toFloat(&converted);
  if (converted)
  {
    switch (row)
    {
    case 0:
      weights.labelShadowConstraint = value;
      break;
    case 1:
      weights.distanceToOldPosition = value;
      break;
    case 2:
      weights.distanceToAnchor = value;
      break;
    case 3:
      weights.favorHorizontalOrVerticalLines = value;
      break;
    case 4:
      weights.connectorShadowConstraint = value;
      break;
    case 5:
      weights.anchorConstraint = value;
      break;
    case 6:
      weights.integralCosts = value;
      break;
    case 7:
      integralCostsWeights.saliency = value;
      break;
    case 8:
      integralCostsWeights.occlusion = value;
      break;
    }
  }

  coordinator->setCostFunctionWeights(weights);
}

void PlacementLabellerModel::toggleVisibility()
{
  isVisible = !isVisible;
  emit isVisibleChanged();
}

void PlacementLabellerModel::simulateHardConstraints()
{
  beginResetModel();

  weights.connectorShadowConstraint = std::numeric_limits<float>::max();
  weights.labelShadowConstraint = std::numeric_limits<float>::max();
  coordinator->setCostFunctionWeights(weights);

  endResetModel();
}

QString PlacementLabellerModel::getWeightNameForRowIndex(int rowIndex) const
{
  switch (rowIndex)
  {
  case 0:
    return "Label shadow";
  case 1:
    return "Distance to old position";
  case 2:
    return "Distance to anchor";
  case 3:
    return "Connector orientation";
  case 4:
    return "Connector shadow";
  case 5:
    return "Anchor constraint";
  case 6:
    return "Integral costs:";
  case 7:
    return "- Saliency";
  case 8:
    return "- Occlusion";
  }

  return "Unknown";
}

float PlacementLabellerModel::getWeightValueForRowIndex(int rowIndex) const
{
  switch (rowIndex)
  {
  case 0:
    return weights.labelShadowConstraint;
  case 1:
    return weights.distanceToOldPosition;
  case 2:
    return weights.distanceToAnchor;
  case 3:
    return weights.favorHorizontalOrVerticalLines;
  case 4:
    return weights.connectorShadowConstraint;
  case 5:
    return weights.anchorConstraint;
  case 6:
    return weights.integralCosts;
  case 7:
    return integralCostsWeights.saliency;
  case 8:
    return integralCostsWeights.occlusion;
  }

  return 0.0f;
}

