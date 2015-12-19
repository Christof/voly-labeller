#include "./application.h"
#include <QQmlContext>
#include <QStateMachine>
#include <QDebug>
#include "./window.h"
#include "./scene.h"
#include "./scene_controller.h"
#include "./nodes.h"
#include "./label_node.h"
#include "./input/invoke_manager.h"
#include "./input/signal_manager.h"
#include "./input/scxml_importer.h"
#include "./mouse_shape_controller.h"
#include "./labeller_model.h"
#include "./labels_model.h"
#include "./labelling/labels.h"
#include "./picking_controller.h"
#include "./forces_visualizer_node.h"
#include "./placement/labeller.h"
#include "./default_scene_creator.h"
#include "./texture_mapper_manager.h"
#include "./texture_mapper_manager_controller.h"
#include "./utils/memory.h"

Application::Application(int argc, char **argv) : application(argc, argv)
{
}

Application::~Application()
{
}

void onLabelChangedUpdateLabelNodes(std::shared_ptr<Nodes> nodes,
                                    Labels::Action action, const Label &label)
{
  auto labelNodes = nodes->getLabelNodes();
  auto labelNode = std::find_if(labelNodes.begin(), labelNodes.end(),
                                [label](std::shared_ptr<LabelNode> labelNode)
                                {
    return labelNode->label.id == label.id;
  });

  if (labelNode == labelNodes.end())
  {
    nodes->addNode(std::make_shared<LabelNode>(label));
  }
  else if (action == Labels::Action::Delete)
  {
    nodes->removeNode(*labelNode);
  }
  else
  {
    (*labelNode)->label = label;
  }
};

int Application::execute()
{
  qInfo() << "Application start";

  auto invokeManager = std::shared_ptr<InvokeManager>(new InvokeManager());
  auto nodes = std::make_shared<Nodes>();
  auto labels = std::make_shared<Labels>();
  auto forcesLabeller = std::make_shared<Forces::Labeller>(labels);
  auto placementLabeller = std::make_shared<Placement::Labeller>(labels);

  const int postProcessingTextureSize = 512;
  auto textureMapperManager =
      std::make_shared<TextureMapperManager>(postProcessingTextureSize);
  auto textureMapperManagerController =
      std::make_unique<TextureMapperManagerController>(textureMapperManager);

  auto scene =
      std::make_shared<Scene>(invokeManager, nodes, labels, forcesLabeller,
                              placementLabeller, textureMapperManager);
  SceneController sceneController(scene);

  auto unsubscribeLabelChanges = labels->subscribe(
      std::bind(&onLabelChangedUpdateLabelNodes, nodes, std::placeholders::_1,
                std::placeholders::_2));

  std::unique_ptr<Window> window = std::unique_ptr<Window>(new Window(scene));
  window->setResizeMode(QQuickView::SizeRootObjectToView);
  window->rootContext()->setContextProperty("window", window.get());
  window->rootContext()->setContextProperty("nodes", nodes.get());
  window->rootContext()->setContextProperty(
      "bufferTextures", textureMapperManagerController.get());
  window->rootContext()->setContextProperty("scene", &sceneController);

  MouseShapeController mouseShapeController;
  PickingController pickingController(scene);

  LabellerModel labellerModel(forcesLabeller);
  labellerModel.connect(&labellerModel, &LabellerModel::isVisibleChanged,
                        [&labellerModel, &nodes, &forcesLabeller]()
                        {
    if (labellerModel.getIsVisible())
      nodes->addForcesVisualizerNode(
          std::make_shared<ForcesVisualizerNode>(forcesLabeller));
    else
      nodes->removeForcesVisualizerNode();
  });
  window->rootContext()->setContextProperty("labeller", &labellerModel);

  LabelsModel labelsModel(labels, pickingController);
  window->rootContext()->setContextProperty("labels", &labelsModel);
  window->setSource(QUrl("qrc:ui.qml"));

  forcesLabeller->resize(window->size().width(), window->size().height());

  QObject::connect(nodes.get(), &Nodes::nodesChanged,
                   [nodes, labels](std::shared_ptr<Node> node)
                   {
    std::shared_ptr<LabelNode> labelNode =
        std::dynamic_pointer_cast<LabelNode>(node);
    if (labelNode.get())
      labels->add(labelNode->label);
  });

  DefaultSceneCreator sceneCreator(nodes, labels);
  sceneCreator.create();

  auto signalManager = std::shared_ptr<SignalManager>(new SignalManager());
  ScxmlImporter importer(QUrl::fromLocalFile("config/states.xml"),
                         invokeManager, signalManager);

  invokeManager->addHandler(window.get());
  invokeManager->addHandler("mouseShape", &mouseShapeController);
  invokeManager->addHandler("picking", &pickingController);
  signalManager->addSender("KeyboardEventSender", window.get());
  signalManager->addSender("labels", &labelsModel);

  auto stateMachine = importer.import();

  // just for printCurrentState slot for debugging
  window->stateMachine = stateMachine;

  stateMachine->start();

  window->show();

  auto resultCode = application.exec();

  unsubscribeLabelChanges();

  return resultCode;
}
