import QtQuick 2.7
import QtQuick.Controls 1.4

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
    x: 570; y: 250
    spacing: 4
    width: 430
    height: 280
    visible: cameraPositions.isVisible

    TableView {
      id: cameraPositionsTableView
      TableViewColumn {
        role: "name"
        title: "Name"
        width: 380
        delegate: cameraPositionNameDelegate
      }
      onDoubleClicked: {
        cameraPositions.moveTo(row);
      }

      width: 405
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
    Row {
      spacing: 10
      Button {
        text: "Move to position"
        onClicked: {
          if (cameraPositions) cameraPositions.moveTo(cameraPositionsTableView.currentRow);
        }
      }
      Button {
        text: "Set position"
        onClicked: {
          if (cameraPositions) cameraPositions.setTo(cameraPositionsTableView.currentRow);
        }
      }
    }
    Row {
      spacing: 10
      Button {
          text: "Record video"
          onClicked: {
              if (automation) automation.startVideo(positions.text);
          }
      }

      TextField {
          id: positions
          width: 310
          height: 24
          placeholderText: qsTr("Positions separated by ','")
      }
    }
  }
}
