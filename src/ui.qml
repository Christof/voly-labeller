import QtQuick 2.1
import QtQuick.Controls 1.2
import QtQuick.Dialogs 1.0

FocusScope
{
  id: scope
  x: root.x; y: root.y
  width: root.width; height: root.height
  onActiveFocusChanged: { window.uiFocusChanged(scope.activeFocus); }

  Item {
    id: root
    x: 0
    y: 0
    width: 1024
    height: 1024

    function toggleVisibility() {
        root.visible = !root.visible;
    }

    Item {
      id: menuWrapper
      anchors.fill: parent

      MenuBar {
        id: myTopMenu
        Menu {
          title: "File"
          MenuItem {
            text: "Open"
            shortcut: "Ctrl+o"
            onTriggered: addSceneNodesFromDialog.open();
          }
          MenuItem {
            text: "Import Mesh"
            shortcut: "Ctrl+i"
            onTriggered: importDialog.open();
          }
          MenuItem {
            text: "Import Volume"
            shortcut: "Ctrl+l"
            onTriggered: importVolumeDialog.open();
          }
          MenuItem {
            text: "Save"
            shortcut: "Ctrl+s"
            onTriggered: saveSceneDialog.open();
          }
          MenuItem {
            text: "Reset scene"
            shortcut: "Ctrl+r"
            onTriggered: {
              labels.clear();
              nodes.clear();
            }
          }
          MenuItem {
            text: "Exit"
            shortcut: "Esc"
            onTriggered: Qt.quit();
          }
        }
        Menu {
          title: "View"
          MenuItem {
            text: "Hide user interface"
            shortcut: "F1"
            onTriggered: toggleVisibility();
          }
          MenuItem {
            text: "Toggle bounding volumes"
            shortcut: "F2"
            onTriggered: nodes.toggleBoundingVolumes();
          }
          MenuItem {
            text: "Toggle forces info"
            shortcut: "F3"
            onTriggered: labeller.toggleForcesVisibility();
          }
          MenuItem {
            text: "Toggle labels info"
            shortcut: "F4"
            onTriggered: labels.toggleLabelsInfoVisbility();
          }
          MenuItem {
            text: "Toggle buffer views"
            shortcut: "F5"
            onTriggered: scene.toggleBufferViews();
          }
          MenuItem {
            text: "Toggle placement info"
            shortcut: "F6"
            onTriggered: placement.toggleVisibility();
          }
          MenuItem {
            text: "Toggle constraint overlay"
            shortcut: "F7"
            onTriggered: scene.toggleConstraintOverlay();
          }
          MenuItem {
            text: "Toggle fullscreen"
            shortcut: "F11"
            onTriggered: window.toggleFullscreen();
          }
        }
        Menu {
          title: "Simulation"
          MenuItem {
            text: "Toggle label update"
            shortcut: "Space"
            onTriggered: labeller.toggleUpdatePositions();
          }
          MenuItem {
            text: "Toggle forces"
            shortcut: "F8"
            onTriggered: labelling.toggleForces();
          }
          MenuItem {
            text: "Toggle optimization on idle"
            shortcut: "F9"
            onTriggered: labelling.toggleOptimizeOnIdle();
          }
        }
        Menu {
          title: "Video recorder"
          MenuItem {
            text: "Start new recording"
            shortcut: "C"
            onTriggered: videoRecorder.startNewVideo();
          }
          MenuItem {
            text: videoRecorder.toggleRecordingText
            enabled: videoRecorder.canToggleRecording
            shortcut: "T"
            onTriggered: videoRecorder.toggleRecording();
          }
        }
        Menu {
          title: "Debug"
          MenuItem {
            text: "Save occlusion"
            shortcut: "O"
            onTriggered: labelling.saveOcclusion();
          }
          MenuItem {
            text: "Save distance transform"
            shortcut: "I"
            onTriggered: bufferTextures.saveDistanceTransform();
          }
          MenuItem {
            text: "Save apollonius"
            shortcut: "P"
            onTriggered: bufferTextures.saveApollonius();
          }
        }
      }

      MouseArea {
          id: mouseArea1
          anchors.fill: parent
          onClicked: { scope.focus = false; }
      }

      FileDialog {
          id: addSceneNodesFromDialog
          title: "Please choose a scene file"
          nameFilters: [ "Xml files (*.xml)" ]
          onAccepted: {
              nodes.addSceneNodesFrom(fileUrl);
          }
      }

      FileDialog {
          id: importDialog
          title: "Please choose a file to import"
          nameFilters: [ "Collada files (*.dae)", , "All files (*)" ]
          onAccepted: {
              nodes.importMeshFrom(fileUrl);
          }
      }

      FileDialog {
          id: importVolumeDialog
          title: "Please choose a volume to import"
          nameFilters: [ "Volume files (*.mhd *.mha *.img)", "All files (*)" ]
          onAccepted: {
              nodes.setVolumeToImport(fileUrl);
              importTransferFunctionDialog.open();
          }
      }

      FileDialog {
          id: importTransferFunctionDialog
          title: "Please choose a transfer function"
          nameFilters: [ "Transfer function gradient (*.gra)", "All files (*)" ]
          onAccepted: {
              nodes.importVolume(fileUrl);
          }
      }

      FileDialog {
          id: saveSceneDialog
          selectExisting: false
          title: "Please choose a scene file"
          nameFilters: [ "Xml files (*.xml)" ]
          onAccepted: {
              nodes.saveSceneTo(fileUrl);
          }
      }

      states: State {
        name: "hasMenuBar"
        when: myTopMenu && !myTopMenu.__isNative

        ParentChange {
          target: myTopMenu.__contentItem
          parent: root
        }

        PropertyChanges {
          target: myTopMenu.__contentItem
          x: 0
          y: 0
          width: menuWrapper.width
        }
      }

      Item {
        Loader {
          source: "ForcesTableView.qml"
        }
      }

      Item {
        Loader {
          source: "LabelsTableView.qml"
        }
      }

      Item {
        Loader {
          source: "PlacementWeightsTableView.qml"
        }
      }

      Label {
          id: averageFrameTimeLabel
          text: (window.averageFrameTime * 1000).toFixed(2) + " ms";
          anchors.right: parent.right
          anchors.rightMargin: 8
          anchors.top: parent.top
          anchors.topMargin: 28
          horizontalAlignment: Text.AlignRight
      }
    }
  }
}
