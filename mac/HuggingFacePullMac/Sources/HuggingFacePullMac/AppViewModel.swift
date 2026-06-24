import Foundation
import SwiftUI
import HuggingFacePullMacCore

@MainActor
final class AppViewModel: ObservableObject {
    private static let initialPythonExecutable = defaultPythonExecutable()

    @Published var state: AppState?
    @Published var searchResults: [SearchResult] = []
    @Published var fileResults: [HubFile] = []
    @Published var cleanupResult: CleanupResult?
    @Published var selectedQueueItemID: String?
    @Published var notice: Notice?
    @Published var isSearching = false
    @Published var isInspectingFiles = false
    @Published var isCleaning = false
    @Published var backendHealth: BackendHealth = .starting

    @AppStorage("backendURL") var backendURL = "http://127.0.0.1:8019"
    @AppStorage("autoLaunchBackend") var autoLaunchBackend = true
    @AppStorage("pythonExecutable") var pythonExecutable = initialPythonExecutable

    private let backend = BackendProcessManager()
    private var pollTask: Task<Void, Never>?

    var backendIsRunning: Bool {
        backend.isRunning
    }

    var selectedQueueItem: QueueItem? {
        guard let selectedQueueItemID else {
            return state?.items.first(where: { $0.status == .running }) ?? state?.items.last
        }
        return state?.items.first(where: { $0.id == selectedQueueItemID })
    }

    var apiClient: APIClient {
        APIClient(baseURL: URL(string: backendURL) ?? URL(string: "http://127.0.0.1:8019")!)
    }

    func start() async {
        if autoLaunchBackend {
            launchBackend()
        }
        pollTask?.cancel()
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                await self?.refresh(showErrors: false)
                try? await Task.sleep(for: .seconds(1))
            }
        }
    }

    func launchBackend() {
        backend.start(configuration: BackendLaunchConfiguration(
            repositoryRoot: Self.defaultRepositoryRoot(),
            port: URL(string: backendURL)?.port ?? 8019,
            pythonExecutable: pythonExecutable
        ))
    }

    func stopBackend() {
        backend.stop()
    }

    func shutdown() {
        pollTask?.cancel()
        pollTask = nil
        backend.stop()
    }

    func refresh(showErrors: Bool = true) async {
        do {
            state = try await apiClient.fetchState()
            backendHealth = .available
        } catch {
            backendHealth = .unavailable
            if showErrors {
                show(error)
            }
        }
    }

    func search(query: String) async {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            searchResults = []
            return
        }
        isSearching = true
        defer { isSearching = false }
        do {
            let response = try await apiClient.search(query: trimmed)
            searchResults = response.results
            if response.available == false {
                notice = Notice(message: response.error ?? "Search is unavailable.", isError: true)
            }
        } catch {
            show(error)
        }
    }

    func inspectFiles(repoID: String, revision: String, repoType: RepoType) async {
        guard !repoID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            notice = Notice(message: "Enter a Hugging Face repo ID before inspecting files.", isError: true)
            return
        }
        isInspectingFiles = true
        defer { isInspectingFiles = false }
        do {
            let response = try await apiClient.files(
                repoID: repoID,
                revision: revision.isEmpty ? "main" : revision,
                repoType: repoType
            )
            fileResults = response.files
        } catch {
            show(error)
        }
    }

    func queue(repoID: String, revision: String, repoType: RepoType, allowPatterns: String, ignorePatterns: String) async {
        let trimmed = repoID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            notice = Notice(message: "HF repo ID is required.", isError: true)
            return
        }
        do {
            let item = try await apiClient.queue(QueueRequest(
                repoID: trimmed,
                revision: revision.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "main" : revision,
                repoType: repoType,
                allowPatterns: Self.patterns(from: allowPatterns),
                ignorePatterns: Self.patterns(from: ignorePatterns)
            ))
            selectedQueueItemID = item.id
            notice = Notice(message: "Queued \(item.repoID)", isError: false)
            await refresh()
        } catch {
            show(error)
        }
    }

    func startQueue() async {
        await performStateAction { try await apiClient.startQueue() }
    }

    func pauseQueue() async {
        await performStateAction { try await apiClient.pauseQueue() }
    }

    func stopAfterCurrentFile() async {
        await performStateAction { try await apiClient.stopAfterCurrentFile() }
    }

    func retry(_ item: QueueItem) async {
        do {
            _ = try await apiClient.retry(itemID: item.id)
            await refresh()
        } catch {
            show(error)
        }
    }

    func remove(_ item: QueueItem) async {
        do {
            try await apiClient.remove(itemID: item.id)
            if selectedQueueItemID == item.id {
                selectedQueueItemID = nil
            }
            await refresh()
        } catch {
            show(error)
        }
    }

    func removeInstalled(_ model: InstalledModel) async {
        do {
            try await apiClient.removeInstalled(InstalledRemoveRequest(
                repoID: model.repoID,
                revision: model.revision,
                repoType: model.repoType
            ))
            notice = Notice(message: "Deleted \(model.repoID)", isError: false)
            await refresh()
        } catch {
            show(error)
        }
    }

    func scanCleanup(includePartials: Bool, olderThanDays: Int) async {
        await cleanup(delete: false, includePartials: includePartials, olderThanDays: olderThanDays)
    }

    func deleteCleanup(includePartials: Bool, olderThanDays: Int) async {
        await cleanup(delete: true, includePartials: includePartials, olderThanDays: olderThanDays)
    }

    private func cleanup(delete: Bool, includePartials: Bool, olderThanDays: Int) async {
        isCleaning = true
        defer { isCleaning = false }
        do {
            cleanupResult = delete
                ? try await apiClient.deleteCleanup(includePartials: includePartials, olderThanDays: olderThanDays)
                : try await apiClient.scanCleanup(includePartials: includePartials, olderThanDays: olderThanDays)
            notice = Notice(message: delete ? "Cleanup delete finished." : "Cleanup scan finished.", isError: false)
            await refresh()
        } catch {
            show(error)
        }
    }

    private func performStateAction(_ action: () async throws -> AppState) async {
        do {
            state = try await action()
        } catch {
            show(error)
        }
    }

    private func show(_ error: Error) {
        notice = Notice(message: error.localizedDescription, isError: true)
    }

    private static func patterns(from text: String) -> [String] {
        text.split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    private static func defaultRepositoryRoot() -> URL {
        var url = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        for _ in 0..<5 {
            if FileManager.default.fileExists(atPath: url.appendingPathComponent("pyproject.toml").path) {
                return url
            }
            url.deleteLastPathComponent()
        }
        return URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    }

    private static func defaultPythonExecutable() -> String {
        let root = defaultRepositoryRoot()
        let venvPython = root.appendingPathComponent(".venv/bin/python").path
        if FileManager.default.isExecutableFile(atPath: venvPython) {
            return venvPython
        }
        return "python3"
    }
}

struct Notice: Identifiable, Equatable {
    let id = UUID()
    let message: String
    let isError: Bool
}

enum BackendHealth: String {
    case starting = "Starting"
    case available = "Connected"
    case unavailable = "Offline"
}
