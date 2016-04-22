import QtQuick 2.1
import QtQuick.Controls 1.2
import QtQuick.Dialogs 1.0

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

    Component {
      id: forceDelegate
      Item {
        width: 400; height: 30
        Row {
          CheckBox {
            checked: enabled
          }
          Text {
            text: name
            width: 200
            focus: true
          }
          TextEdit { text: weight }
        }
      }
    }

    Component {
      id: checkBoxDelegate

      Item {
        CheckBox {
          anchors.fill: parent
          checked: styleData.value
          onCheckedChanged: {
            if (labeller) labeller.changeEnabled(styleData.row, checked);
          }
        }
      }
    }

    Component {
      id: textDelegate

      Item {
        Row {
          TextInput {
            id: weightInput
            maximumLength: 6
            width: 50
            validator: DoubleValidator {
              bottom: 0
              top: 10
            }
            text: styleData.value
            color: model ? model.forceColor : "black"
            onTextChanged: {
              weightSlider.value = text;
              if (labeller) labeller.changeWeight(styleData.row, text);
            }
          }
          Slider {
            id: weightSlider
            width: 200
            minimumValue: 0
            maximumValue: 10
            value: styleData.value
            onValueChanged: {
              weightInput.text = value;
              if (labeller) labeller.changeWeight(styleData.row, value);
            }
          }
        }
      }
    }

    Component {
      id: nameDelegate

      Item {
        Text {
          text: styleData.value
          color: model ? model.forceColor : "black"
        }
      }
    }

    TableView {
      visible: labeller.isVisible
      TableViewColumn {
        role: "enabled"
        title: "Enabled"
        width: 100
        delegate: checkBoxDelegate
      }
      TableViewColumn {
        role: "name"
        title: "Name"
        width: 200
        delegate: nameDelegate
      }
      TableViewColumn {
        role: "weight"
        title: "Weight"
        width: 250
        delegate: textDelegate
      }
      x: 10; y: 36
      width: 564
      height: 150
      model: labeller
      focus: true
      clip: true
    }

    Component {
      id: labelNameDelegate

      Item {
        TextInput {
          anchors.fill: parent
          text: styleData.value
          onTextChanged: {
            if (labels) labels.changeText(styleData.row, text);
          }
        }
      }
    }
    Component {
      id: sizeYDelegate

      Item {
        TextInput {
          anchors.fill: parent
          maximumLength: 5
          text: styleData.value
          onTextChanged: {
            if (labels) labels.changeSizeY(styleData.row, text);
          }
        }
      }
    }
    Component {
      id: sizeXDelegate

      Item {
        TextInput {
          anchors.fill: parent
          maximumLength: 5
          text: styleData.value
          onTextChanged: {
            if (labels) labels.changeSizeX(styleData.row, text);
          }
        }
      }
    }
    Component {
      id: anchorDelegate

      Item {
        Button {
          anchors.fill: parent
          text: "Pick"
          onClicked: {
            if (labels) labels.pick(styleData.row);
          }
        }
      }
    }

    Column {
      id: labelsItem
      x: 10
      y: 192
      width: 441
      height: 197
      visible: labels.isVisible
      Row {
        Button {
          text: "Add Label"
          onClicked: {
            if (labels) labels.addLabel();
          }
        }
        Button {
          text: "Delete Label"
          onClicked: {
            if (labels) labels.deleteLabel(labelsView.currentRow);
          }
        }
      }


      TableView {
        id: labelsView
        TableViewColumn {
          role: "name"
          title: "Name"
          width: 150
          delegate: labelNameDelegate
        }
        TableViewColumn {
          role: "sizeX"
          title: "Width"
          width: 100
          delegate: sizeXDelegate
        }
        TableViewColumn {
          role: "sizeY"
          title: "Height"
          width: 100
          delegate: sizeYDelegate
        }
        TableViewColumn {
          title: "Anchor"
          width: 46
          delegate: anchorDelegate
        }
        width: 417
        height: 190
        model: labels
        focus: true
        clip: true
      }
    }

      Component {
        id: weightDelegate
        Item {
          width: 400; height: 30
          Row {
            Text {
              text: name
              width: 200
              focus: true
            }
            TextEdit { text: weight }
          }
        }
      }

      Component {
        id: costFunctionWeightDelegate

        Item {
          TextInput {
            anchors.fill: parent
            maximumLength: 6
            text: styleData.value
            onTextChanged: {
              if (placement) placement.changeWeight(styleData.row, text);
            }
          }
        }
      }

      TableView {
        visible: placement.isVisible
        TableViewColumn {
          role: "name"
          title: "Name"
          width: 200
          delegate: 
            Item {
              Text {
                text: styleData.value
              }
            }
        }
        TableViewColumn {
          role: "weight"
          title: "Weight"
          width: 98
          delegate: costFunctionWeightDelegate
        }
        x: 683; y: 51
        width: 333
        height: 170
        model: placement
        focus: true
        clip: true
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
