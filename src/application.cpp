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
#include "./labelling_coordinator.h"
#include "./picking_controller.h"
#include "./forces_visualizer_node.h"
#include "./default_scene_creator.h"
#include "./texture_mapper_manager.h"
#include "./texture_mapper_manager_controller.h"
#include "./utils/memory.h"
#include "./utils/path_helper.h"

const int LAYER_COUNT = 4;

Application::Application(int &argc, char **argv) : application(argc, argv)
{
  invokeManager = std::make_shared<InvokeManager>();

  nodes = std::make_shared<Nodes>();

  labels = std::make_shared<Labels>();
  forcesLabeller = std::make_shared<Forces::Labeller>(labels);
  auto labellingCoordinator = std::make_shared<LabellingCoordinator>(
      LAYER_COUNT, forcesLabeller, labels, nodes);

  const int postProcessingTextureSize = 512;
  textureMapperManager =
      std::make_shared<TextureMapperManager>(postProcessingTextureSize);
  textureMapperManagerController =
      std::make_unique<TextureMapperManagerController>(textureMapperManager);
  scene = std::make_shared<Scene>(LAYER_COUNT, invokeManager, nodes, labels,
                                  labellingCoordinator, textureMapperManager);

  window = std::make_unique<Window>(scene);
  sceneController = std::make_unique<SceneController>(scene);
  labellerModel = std::make_unique<LabellerModel>(forcesLabeller);
  mouseShapeController = std::make_unique<MouseShapeController>();
  pickingController = std::make_shared<PickingController>(scene);
  labelsModel = std::make_unique<LabelsModel>(labels, pickingController);
}

Application::~Application()
{
}

int Application::execute()
{
  setupCommandLineParser();
  parser.process(application);
  qInfo() << "Application start";

  auto unsubscribeLabelChanges = labels->subscribe(
      std::bind(&Application::onLabelChangedUpdateLabelNodes, this,
                std::placeholders::_1, std::placeholders::_2));

  setupWindow();

  connect(labellerModel.get(), &LabellerModel::isVisibleChanged, this,
          &Application::onFocesLabellerModelIsVisibleChanged);

  window->setSource(QUrl("qrc:ui.qml"));

  forcesLabeller->resize(window->size().width(), window->size().height());

  QObject::connect(nodes.get(), &Nodes::nodesChanged, this,
                   &Application::onNodesChanged);

  if (parser.positionalArguments().size())
  {
    auto filename = absolutePathToProjectRelativePath(
        QDir(parser.positionalArguments()[0]).absolutePath());
    qInfo() << "import scene:" << filename;
    nodes->addSceneNodesFrom(filename);
  }
  else
  {
    DefaultSceneCreator sceneCreator(nodes, labels);
    sceneCreator.create();
  }

  createAndStartStateMachine();

  window->show();

  auto resultCode = application.exec();

  unsubscribeLabelChanges();

  return resultCode;
}

void Application::setupCommandLineParser()
{
  QGuiApplication::setApplicationName("voly-labeller");
  QGuiApplication::setApplicationVersion("0.1");

  parser.setApplicationDescription(
      "Multiple labelling implementations for volume rendered medical data");
  parser.addHelpOption();
  parser.addVersionOption();
  parser.addPositionalArgument(
      "scene", QCoreApplication::translate("main", "Scene file to load."));
}

void Application::setupWindow()
{
  window->setResizeMode(QQuickView::SizeRootObjectToView);
  auto context = window->rootContext();
  context->setContextProperty("window", window.get());
  context->setContextProperty("nodes", nodes.get());
  context->setContextProperty("bufferTextures",
                              textureMapperManagerController.get());
  context->setContextProperty("scene", sceneController.get());
  context->setContextProperty("labeller", labellerModel.get());
  context->setContextProperty("labels", labelsModel.get());
}

void Application::createAndStartStateMachine()
{
  auto signalManager = std::shared_ptr<SignalManager>(new SignalManager());
  ScxmlImporter importer(QUrl::fromLocalFile("config/states.xml"),
                         invokeManager, signalManager);

  invokeManager->addHandler(window.get());
  invokeManager->addHandler("mouseShape", mouseShapeController.get());
  invokeManager->addHandler("picking", pickingController.get());
  invokeManager->addHandler("scene", sceneController.get());
  signalManager->addSender("KeyboardEventSender", window.get());
  signalManager->addSender("labels", labelsModel.get());

  stateMachine = importer.import();

  // just for printCurrentState slot for debugging
  window->stateMachine = stateMachine;

  stateMachine->start();
}

void Application::onNodesChanged(std::shared_ptr<Node> node)
{
  std::shared_ptr<LabelNode> labelNode =
      std::dynamic_pointer_cast<LabelNode>(node);
  if (labelNode.get())
    labels->add(labelNode->label);
}

void Application::onLabelChangedUpdateLabelNodes(Labels::Action action,
                                                 const Label &label)
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

void Application::onFocesLabellerModelIsVisibleChanged()
{
  if (labellerModel->getIsVisible())
  {
    nodes->addForcesVisualizerNode(
        std::make_shared<ForcesVisualizerNode>(forcesLabeller));
  }
  else
  {
    nodes->removeForcesVisualizerNode();
  }
}

