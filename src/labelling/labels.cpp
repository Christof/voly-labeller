#include "./labels.h"
#include <vector>

std::function<void()>
Labels::subscribe(std::function<void(Action action, const Label &)> subscriber)
{
  int erasePosition = subscribers.size();
  subscribers.push_back(subscriber);

  return [this, erasePosition]()
  {
    subscribers.erase(subscribers.begin() + erasePosition);
  };
}

void Labels::add(Label label)
{
  LabelsContainer::add(label);
  notify(Action::Add, label);
}

void Labels::update(Label label)
{
  bool notifyChanges = labels[label.id] != label;

  labels[label.id] = label;

  if (notifyChanges)
    notify(Action::Update, label);
}

void Labels::remove(Label label)
{
  labels.erase(labels.find(label.id));
  notify(Action::Delete, label);
}

void Labels::notify(Action action, const Label &label)
{
  for (auto &subscriber : subscribers)
    subscriber(action, label);
}

