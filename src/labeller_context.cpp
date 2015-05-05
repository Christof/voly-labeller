#include "./labeller_context.h"

LabellerContext::LabellerContext(std::shared_ptr<Forces::Labeller> labeller)
  : labeller(labeller)
{
}

QHash<int, QByteArray> LabellerContext::roleNames() const
{
  QHash<int, QByteArray> roles;
  roles[NameRole] = "name";
  roles[EnabledRole] = "enabled";
  roles[WeightRole] = "weight";

  return roles;
}

int LabellerContext::rowCount(const QModelIndex &parent) const
{
  Q_UNUSED(parent);
  return labeller->forces.size();
}

QVariant LabellerContext::data(const QModelIndex &index, int role) const
{
  if (index.row() < 0 ||
      index.row() >= static_cast<int>(labeller->forces.size()))
    return QVariant();

  auto  &force = labeller->forces[index.row()];
  switch(role)
  {
    case NameRole:
      return force->name.c_str();
    case EnabledRole:
      return force->isEnabled;
    case WeightRole:
      return force->weight;
  }
  return QVariant();
}
