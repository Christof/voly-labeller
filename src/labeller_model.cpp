#include "./labeller_model.h"
#include <QColor>
#include <Eigen/Core>

LabellerModel::LabellerModel(std::shared_ptr<Forces::Labeller> labeller)
  : labeller(labeller)
{
}

QHash<int, QByteArray> LabellerModel::roleNames() const
{
  QHash<int, QByteArray> roles;
  roles[NameRole] = "name";
  roles[EnabledRole] = "enabled";
  roles[WeightRole] = "weight";
  roles[ColorRole] = "forceColor";

  return roles;
}

int LabellerModel::rowCount(const QModelIndex &parent) const
{
  Q_UNUSED(parent);
  return static_cast<int>(labeller->forces.size());
}

int LabellerModel::columnCount(const QModelIndex &parent) const
{
  Q_UNUSED(parent);
  return 3;
}

QVariant LabellerModel::data(const QModelIndex &index, int role) const
{
  if (index.row() < 0 ||
      index.row() >= static_cast<int>(labeller->forces.size()))
    return QVariant();

  auto &force = labeller->forces[index.row()];
  switch (role)
  {
  case NameRole:
    return QVariant::fromValue(QString(force->name.c_str()));
  case EnabledRole:
    return QVariant::fromValue(force->isEnabled);
  case WeightRole:
    return QVariant::fromValue(force->weight);
  case ColorRole:
    auto color = force->color;
    return QVariant::fromValue(
        QColor::fromRgbF(color.x(), color.y(), color.z()));
  }
  return QVariant();
}

Qt::ItemFlags LabellerModel::flags(const QModelIndex &index) const
{
  if (!index.isValid())
    return Qt::ItemIsEnabled;

  return QAbstractItemModel::flags(index) | Qt::ItemFlag::ItemIsEditable |
         Qt::ItemIsUserCheckable;
}

bool LabellerModel::getIsVisible() const
{
  return isVisible;
}

void LabellerModel::changeEnabled(int row, QVariant newValue)
{
  labeller->forces[row]->isEnabled = newValue.toBool();
}

void LabellerModel::changeWeight(int row, QVariant newValue)
{
  bool converted = false;
  auto value = newValue.toFloat(&converted);
  if (converted)
    labeller->forces[row]->weight = value;
}

void LabellerModel::toggleUpdatePositions()
{
  labeller->updatePositions = !labeller->updatePositions;
}

void LabellerModel::toggleForcesVisibility()
{
  isVisible = !isVisible;
  emit isVisibleChanged();
}

