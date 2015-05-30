import QtQuick 2.1
import QtQuick.Controls 1.2
import QtQuick.Dialogs 1.0

Item {
  id: root
  x: 0
  y: 0
  width: 1280
  height: 720

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
          text: "Import"
          shortcut: "Ctrl+i"
          onTriggered: importDialog.open();
        }
        MenuItem {
          text: "Save"
          shortcut: "Ctrl+s"
          onTriggered: saveSceneDialog.open();
        }
        MenuItem {
          text: "Reset scene"
          shortcut: "Ctrl+r"
          onTriggered: nodes.clear();
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
          onTriggered: labeller.toggleForcesVisbility();
        }
        MenuItem {
          text: "Toggle labels info"
          shortcut: "F4"
          onTriggered: labels.toggleLabelsInfoVisbility();
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
        nameFilters: [ "Collada files (*.dae)" ]
        onAccepted: {
            nodes.importFrom(fileUrl);
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
        TextInput {
          anchors.fill: parent
          maximumLength: 6
          text: styleData.value
          color: model ? model.forceColor : "black"
          onTextChanged: {
            if (labeller) labeller.changeWeight(styleData.row, text);
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
        width: 98
        delegate: textDelegate
      }
      x: 10; y: 36
      width: 400
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
      id: lablesItem
      x: 8
      y: 192
      width: 441
      height: 197
      Button {
          x: 10
          text: "Add Label"
          onClicked: {
            if (labels) labels.addLabel();
          }
      }

      TableView {
        visible: labels.isVisible
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
        x: 10; y: 240
        width: 417
        height: 190
        model: labels
        focus: true
        clip: true
      }
    }

  }
}
