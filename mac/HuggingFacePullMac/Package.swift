// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "HuggingFacePullMac",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .library(
            name: "HuggingFacePullMacCore",
            targets: ["HuggingFacePullMacCore"]
        ),
        .executable(
            name: "HuggingFacePullMac",
            targets: ["HuggingFacePullMac"]
        ),
    ],
    targets: [
        .target(
            name: "HuggingFacePullMacCore"
        ),
        .executableTarget(
            name: "HuggingFacePullMac",
            dependencies: ["HuggingFacePullMacCore"]
        ),
        .testTarget(
            name: "HuggingFacePullMacCoreTests",
            dependencies: ["HuggingFacePullMacCore"]
        ),
    ]
)
