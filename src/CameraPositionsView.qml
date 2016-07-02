import QtQuick 2.1
import QtQuick.Controls 1.2

Item {
  Component {
    id: cameraPositionNameDelegate

    Item {
      TextInput {
        anchors.fill: parent
        text: styleData.value
        onTextChanged: {
          if (cameraPositions) cameraPositions.changeName(styleData.row, text);
        }
      }
    }
  }

  Column {
    x: 70; y: 20
    width: 430
    height: 170

    TableView {
      id: cameraPositionsTableView
      TableViewColumn {
        role: "name"
        title: "Name"
        width: 150
        delegate: cameraPositionNameDelegate
      }
      width: 417
      height: 190
      model: cameraPositions
      focus: true
      clip: true
    }

    Row {
      spacing: 10
      Button {
        text: "Save position"
        onClicked: {
          if (cameraPositions) cameraPositions.save();
        }
      }
      Button {
        text: "Delete position"
        onClicked: {
          if (cameraPositions) cameraPositions.deletePosition(cameraPositionsTableView.currentRow);
        }
      }
    }
  }
}
