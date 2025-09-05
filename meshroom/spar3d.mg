{
    "header": {
        "releaseVersion": "2026.1.0-develop",
        "fileVersion": "2.0",
        "nodesVersions": {
            "CameraInit": "12.0",
            "Publish": "1.3",
            "SPAR3D": "1.0"
        },
        "template": true
    },
    "graph": {
        "CameraInit_1": {
            "nodeType": "CameraInit",
            "position": [
                -333,
                -78
            ],
            "inputs": {}
        },
        "Publish_1": {
            "nodeType": "Publish",
            "position": [
                97,
                -59
            ],
            "inputs": {
                "output": "{SPAR3D_1.output}"
            }
        },
        "SPAR3D_1": {
            "nodeType": "SPAR3D",
            "position": [
                -114,
                -59
            ],
            "inputs": {
                "inputImages": "{CameraInit_1.output}",
                "device": "cpu"
            }
        }
    }
}