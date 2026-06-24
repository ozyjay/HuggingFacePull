import Foundation

public struct BackendLaunchConfiguration: Equatable, Sendable {
    public var repositoryRoot: URL
    public var host: String
    public var port: Int
    public var pythonExecutable: String

    public init(
        repositoryRoot: URL,
        host: String = "127.0.0.1",
        port: Int = 8019,
        pythonExecutable: String = "python3"
    ) {
        self.repositoryRoot = repositoryRoot
        self.host = host
        self.port = port
        self.pythonExecutable = pythonExecutable
    }

    public var baseURL: URL {
        URL(string: "http://\(host):\(port)")!
    }
}

@MainActor
public final class BackendProcessManager: ObservableObject {
    @Published public private(set) var isRunning = false
    @Published public private(set) var lastError: String?

    private var process: Process?

    public init() {}

    public func start(configuration: BackendLaunchConfiguration) {
        guard process == nil else {
            return
        }

        let process = Process()
        process.currentDirectoryURL = configuration.repositoryRoot
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        var environment = ProcessInfo.processInfo.environment
        let sourcePath = configuration.repositoryRoot.appendingPathComponent("src").path
        if let existing = environment["PYTHONPATH"], !existing.isEmpty {
            environment["PYTHONPATH"] = "\(sourcePath):\(existing)"
        } else {
            environment["PYTHONPATH"] = sourcePath
        }
        process.environment = environment
        process.arguments = [
            configuration.pythonExecutable,
            "-m",
            "huggingface_pull.web_main",
            "--host",
            configuration.host,
            "--port",
            String(configuration.port),
            "--no-browser",
        ]
        process.terminationHandler = { [weak self] _ in
            Task { @MainActor in
                self?.isRunning = false
                self?.process = nil
            }
        }

        do {
            try process.run()
            self.process = process
            isRunning = true
            lastError = nil
        } catch {
            lastError = error.localizedDescription
            isRunning = false
        }
    }

    public func stop() {
        guard let process else {
            isRunning = false
            return
        }
        if process.isRunning {
            process.terminate()
            process.waitUntilExit()
        }
        self.process = nil
        isRunning = false
    }
}
