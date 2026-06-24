import AppKit
import SwiftUI
import HuggingFacePullMacCore

enum AppSection: String, CaseIterable, Identifiable {
    case search = "Search"
    case queue = "Queue"
    case installed = "Installed"
    case cleanup = "Cleanup"

    var id: String { rawValue }

    var symbol: String {
        switch self {
        case .search: "magnifyingglass"
        case .queue: "arrow.down.circle"
        case .installed: "internaldrive"
        case .cleanup: "trash"
        }
    }
}

struct ContentView: View {
    @EnvironmentObject private var model: AppViewModel
    @State private var selection: AppSection? = .search

    var body: some View {
        NavigationSplitView {
            List(AppSection.allCases, selection: $selection) { section in
                Label(section.rawValue, systemImage: section.symbol)
                    .tag(section)
            }
            .navigationTitle("HuggingFacePull")
            .safeAreaInset(edge: .bottom) {
                BackendStatusView()
                    .padding()
            }
        } detail: {
            switch selection ?? .search {
            case .search:
                SearchView()
            case .queue:
                QueueView()
            case .installed:
                InstalledView()
            case .cleanup:
                CleanupView()
            }
        }
        .toolbar {
            ToolbarItemGroup {
                Button {
                    Task { await model.startQueue() }
                } label: {
                    Label("Start", systemImage: "play.fill")
                }
                .disabled(model.state?.running == true)

                Button {
                    Task { await model.pauseQueue() }
                } label: {
                    Label("Pause", systemImage: "pause.fill")
                }
                .disabled(model.state?.running != true || model.state?.pauseRequested == true)

                Button(role: .destructive) {
                    Task { await model.stopAfterCurrentFile() }
                } label: {
                    Label("Stop", systemImage: "stop.fill")
                }
                .disabled(model.state?.running != true || model.state?.stopAfterFileRequested == true)

                Button {
                    Task { await model.refresh() }
                } label: {
                    Label("Refresh", systemImage: "arrow.clockwise")
                }
            }
        }
        .overlay(alignment: .bottom) {
            if let notice = model.notice {
                NoticeBanner(notice: notice)
                    .padding()
            }
        }
    }
}

struct FocusedTextField<Field: Hashable>: NSViewRepresentable {
    let placeholder: String
    @Binding var text: String
    @Binding var focusedFieldValue: Field?
    let field: Field
    var onSubmit: (() -> Void)?

    func makeNSView(context: Context) -> NSTextField {
        let textField = NSTextField()
        textField.placeholderString = placeholder
        textField.stringValue = text
        textField.isBordered = true
        textField.isBezeled = true
        textField.bezelStyle = .roundedBezel
        textField.delegate = context.coordinator
        textField.setContentHuggingPriority(.defaultLow, for: .horizontal)
        return textField
    }

    func updateNSView(_ textField: NSTextField, context: Context) {
        if textField.stringValue != text {
            textField.stringValue = text
        }

        if focusedFieldValue == field {
            DispatchQueue.main.async {
                guard let window = textField.window else {
                    return
                }
                guard context.coordinator.lastFocusedField != field else {
                    return
                }
                context.coordinator.lastFocusedField = field
                guard window.firstResponder !== textField.currentEditor() else {
                    return
                }
                window.makeFirstResponder(textField)
            }
        } else if context.coordinator.lastFocusedField == field {
            context.coordinator.lastFocusedField = nil
        }
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    final class Coordinator: NSObject, NSTextFieldDelegate {
        var parent: FocusedTextField
        var lastFocusedField: Field?

        init(_ parent: FocusedTextField) {
            self.parent = parent
        }

        func controlTextDidBeginEditing(_ notification: Notification) {
            parent.focusedFieldValue = parent.field
        }

        func controlTextDidChange(_ notification: Notification) {
            guard let textField = notification.object as? NSTextField else {
                return
            }
            parent.text = textField.stringValue
        }

        func controlTextDidEndEditing(_ notification: Notification) {
            guard let textField = notification.object as? NSTextField else {
                return
            }
            parent.text = textField.stringValue
        }

        func control(_ control: NSControl, textView: NSTextView, doCommandBy commandSelector: Selector) -> Bool {
            guard commandSelector == #selector(NSResponder.insertNewline(_:)) else {
                return false
            }
            parent.onSubmit?()
            return true
        }
    }
}

struct BackendStatusView: View {
    @EnvironmentObject private var model: AppViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label(model.backendHealth.rawValue, systemImage: statusSymbol)
                .font(.caption.weight(.semibold))
                .foregroundStyle(statusColour)
            Text(model.state?.libraryDirectory ?? model.backendURL)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(2)
            HStack {
                Button(model.backendIsRunning ? "Stop Backend" : "Launch Backend") {
                    model.backendIsRunning ? model.stopBackend() : model.launchBackend()
                }
                .controlSize(.small)
                Spacer()
            }
        }
    }

    private var statusSymbol: String {
        switch model.backendHealth {
        case .starting: "clock"
        case .available: "checkmark.circle.fill"
        case .unavailable: "exclamationmark.triangle.fill"
        }
    }

    private var statusColour: Color {
        switch model.backendHealth {
        case .starting: .secondary
        case .available: .green
        case .unavailable: .orange
        }
    }
}

struct SearchView: View {
    private enum SearchField: Hashable {
        case query
        case repoID
        case revision
        case allowPatterns
        case ignorePatterns
    }

    @EnvironmentObject private var model: AppViewModel
    @State private var query = ""
    @State private var repoID = ""
    @State private var revision = "main"
    @State private var repoType: RepoType = .model
    @State private var allowPatterns = "*.json, *.safetensors"
    @State private var ignorePatterns = ""
    @FocusState private var focusedField: SearchField?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                SectionHeader(title: "Search Hub repos", detail: "\(model.searchResults.count) results")
                HStack {
                    FocusedTextField(
                        placeholder: "qwen embedding",
                        text: $query,
                        focusedFieldValue: focusBinding,
                        field: .query,
                        onSubmit: { Task { await model.search(query: query) } }
                    )
                        .focused($focusedField, equals: .query)
                    Button {
                        Task { await model.search(query: query) }
                    } label: {
                        Label("Search", systemImage: "magnifyingglass")
                    }
                    .disabled(model.isSearching)
                }

                ForEach(model.searchResults) { result in
                    SearchResultRow(result: result) {
                        repoID = result.repoID
                        focusedField = .repoID
                        Task {
                            await model.queue(
                                repoID: result.repoID,
                                revision: revision,
                                repoType: repoType,
                                allowPatterns: allowPatterns,
                                ignorePatterns: ignorePatterns
                            )
                        }
                    }
                    Divider()
                }

                SectionHeader(title: "Add HF repo ID", detail: "Queued snapshots use the shared Hugging Face cache.")
                Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 12) {
                    GridRow {
                        Text("Repo ID")
                        FocusedTextField(
                            placeholder: "Qwen/Qwen3-Embedding-0.6B",
                            text: $repoID,
                            focusedFieldValue: focusBinding,
                            field: .repoID
                        )
                            .focused($focusedField, equals: .repoID)
                    }
                    GridRow {
                        Text("Revision")
                        FocusedTextField(
                            placeholder: "main",
                            text: $revision,
                            focusedFieldValue: focusBinding,
                            field: .revision
                        )
                            .focused($focusedField, equals: .revision)
                    }
                    GridRow {
                        Text("Repo type")
                        Picker("Repo type", selection: $repoType) {
                            ForEach(RepoType.allCases, id: \.self) { type in
                                Text(type.rawValue).tag(type)
                            }
                        }
                        .labelsHidden()
                    }
                    GridRow {
                        Text("Allow")
                        FocusedTextField(
                            placeholder: "*.safetensors, *.json",
                            text: $allowPatterns,
                            focusedFieldValue: focusBinding,
                            field: .allowPatterns
                        )
                            .focused($focusedField, equals: .allowPatterns)
                    }
                    GridRow {
                        Text("Ignore")
                        FocusedTextField(
                            placeholder: "*.msgpack, *.h5",
                            text: $ignorePatterns,
                            focusedFieldValue: focusBinding,
                            field: .ignorePatterns
                        )
                            .focused($focusedField, equals: .ignorePatterns)
                    }
                }

                HStack {
                    Button {
                        Task {
                            await model.queue(
                                repoID: repoID,
                                revision: revision,
                                repoType: repoType,
                                allowPatterns: allowPatterns,
                                ignorePatterns: ignorePatterns
                            )
                        }
                    } label: {
                        Label("Add to Queue", systemImage: "plus")
                    }
                    Button {
                        Task {
                            await model.inspectFiles(repoID: repoID, revision: revision, repoType: repoType)
                        }
                    } label: {
                        Label("Inspect Files", systemImage: "doc.text.magnifyingglass")
                    }
                    .disabled(model.isInspectingFiles)
                }

                FileList(files: model.fileResults)
            }
            .padding(24)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .navigationTitle("Search")
        .onAppear {
            focusDefaultField()
        }
    }

    private func focusDefaultField() {
        Task { @MainActor in
            try? await Task.sleep(nanoseconds: 150_000_000)
            if focusedField == nil {
                focusedField = .query
            }
        }
    }

    private var focusBinding: Binding<SearchField?> {
        Binding(
            get: { focusedField },
            set: { focusedField = $0 }
        )
    }
}

struct SearchResultRow: View {
    let result: SearchResult
    let add: () -> Void

    var body: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 4) {
                Text(result.repoID)
                    .font(.headline)
                Text([result.pipelineTag, count(result.downloads, "download"), count(result.likes, "like")]
                    .compactMap { $0 }
                    .joined(separator: " | "))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button(action: add) {
                Label("Add", systemImage: "plus")
            }
        }
    }

    private func count(_ value: Int?, _ label: String) -> String? {
        guard let value else { return nil }
        return "\(value) \(label)\(value == 1 ? "" : "s")"
    }
}

struct FileList: View {
    let files: [HubFile]

    var body: some View {
        if !files.isEmpty {
            SectionHeader(title: "Files", detail: "\(files.count) files")
            VStack(spacing: 0) {
                ForEach(files.prefix(80)) { file in
                    HStack {
                        Text(file.displayName)
                            .lineLimit(1)
                        Spacer()
                        Text(DisplayFormat.bytes(file.size))
                            .foregroundStyle(.secondary)
                    }
                    .font(.caption)
                    .padding(.vertical, 4)
                    Divider()
                }
            }
        }
    }
}

struct QueueView: View {
    @EnvironmentObject private var model: AppViewModel

    var body: some View {
        HSplitView {
            VStack(alignment: .leading, spacing: 12) {
                SectionHeader(
                    title: "Queue",
                    detail: "\(model.state?.items.count ?? 0) items | \(runState)"
                )
                List(model.state?.items ?? [], selection: $model.selectedQueueItemID) { item in
                    QueueRow(item: item)
                        .tag(item.id)
                        .contextMenu {
                            if item.status == .failed {
                                Button("Retry") { Task { await model.retry(item) } }
                            }
                            if item.status != .running {
                                Button("Remove", role: .destructive) { Task { await model.remove(item) } }
                            }
                        }
                }
            }
            .padding(24)
            .frame(minWidth: 420, maxWidth: .infinity, maxHeight: .infinity)

            QueueDetailView(item: model.selectedQueueItem)
                .frame(minWidth: 320, maxWidth: .infinity, maxHeight: .infinity)
        }
        .navigationTitle("Queue")
    }

    private var runState: String {
        guard let state = model.state else { return "loading" }
        if state.stopAfterFileRequested { return "stopping after current file" }
        if state.pauseRequested { return state.running ? "pausing after current download" : "paused" }
        return state.running ? "running" : "idle"
    }
}

struct QueueRow: View {
    let item: QueueItem

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(item.repoID)
                    .font(.headline)
                StatusBadge(status: item.status)
                Spacer()
                Text(DisplayFormat.percent(item.progress.overall.percent))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            Text(statusLine)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)
            ProgressView(value: item.progressFraction ?? 0)
            HStack {
                Text(item.revision)
                Text("\(DisplayFormat.bytes(item.progress.overall.downloaded)) / \(item.progress.overall.total.map(DisplayFormat.bytes) ?? "total calculating")")
                Text(DisplayFormat.speed(item.progress.overall.bytesPerSecond))
                Text("ETA \(DisplayFormat.duration(item.progress.overall.etaSeconds))")
            }
            .font(.caption2)
            .foregroundStyle(.secondary)
        }
        .padding(.vertical, 6)
    }

    private var statusLine: String {
        let current = item.progress.currentFile?.displayName ?? "snapshot"
        return "\(item.progress.phase.capitalized) \(current) | updated \(DisplayFormat.timestamp(item.updatedAt))"
    }
}

struct QueueDetailView: View {
    let item: QueueItem?

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            if let item {
                HStack {
                    VStack(alignment: .leading) {
                        Text(item.repoID)
                            .font(.title3.weight(.semibold))
                        Text("\(item.revision) | \(item.repoType.rawValue)")
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    StatusBadge(status: item.status)
                }
                ProgressView(value: item.progressFraction ?? 0)
                DetailGrid(item: item)
                Text("Messages")
                    .font(.headline)
                List(item.messages.indices, id: \.self) { index in
                    Text(item.messages[index].text ?? "message")
                        .font(.caption)
                }
                if let error = item.error {
                    Text(error)
                        .foregroundStyle(.red)
                }
            } else {
                EmptyStateView(
                    title: "No Queue Item",
                    systemImage: "arrow.down.circle",
                    detail: "Select a queue row to inspect messages and current file."
                )
            }
        }
        .padding(24)
        .navigationTitle("Detail")
    }
}

struct DetailGrid: View {
    let item: QueueItem

    var body: some View {
        Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 8) {
            detail("Phase", item.progress.phase)
            detail("Overall", "\(DisplayFormat.bytes(item.progress.overall.downloaded)) / \(item.progress.overall.total.map(DisplayFormat.bytes) ?? "total calculating")")
            detail("Percent", DisplayFormat.percent(item.progress.overall.percent))
            detail("Speed", DisplayFormat.speed(item.progress.overall.bytesPerSecond))
            detail("ETA", DisplayFormat.duration(item.progress.overall.etaSeconds))
            detail("Current file", item.progress.currentFile?.displayName ?? "None")
            detail("Last update", DisplayFormat.timestamp(item.updatedAt))
        }
        .font(.caption)
    }

    private func detail(_ title: String, _ value: String) -> some View {
        GridRow {
            Text(title).foregroundStyle(.secondary)
            Text(value).textSelection(.enabled)
        }
    }
}

struct InstalledView: View {
    @EnvironmentObject private var model: AppViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            SectionHeader(title: "Installed snapshots", detail: "\(model.state?.installedModels.count ?? 0) snapshots")
            List(model.state?.installedModels ?? []) { installed in
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(installed.repoID)
                            .font(.headline)
                        Text("\(installed.revision) | \(installed.formattedSize) | \(installed.snapshotPath ?? "")")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                    Spacer()
                    Button(role: .destructive) {
                        Task { await model.removeInstalled(installed) }
                    } label: {
                        Label("Delete Record", systemImage: "trash")
                    }
                }
                .padding(.vertical, 4)
            }
        }
        .padding(24)
        .navigationTitle("Installed")
    }
}

struct CleanupView: View {
    @EnvironmentObject private var model: AppViewModel
    @State private var includePartials = true
    @State private var olderThanDays = 0

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            SectionHeader(title: "Cleanup", detail: "Stale partial file scan")
            Toggle("Include app and HF cache partial files", isOn: $includePartials)
            Stepper("Older than \(olderThanDays) days", value: $olderThanDays, in: 0...365)
            HStack {
                Button {
                    Task { await model.scanCleanup(includePartials: includePartials, olderThanDays: olderThanDays) }
                } label: {
                    Label("Scan", systemImage: "magnifyingglass")
                }
                Button(role: .destructive) {
                    Task { await model.deleteCleanup(includePartials: includePartials, olderThanDays: olderThanDays) }
                } label: {
                    Label("Delete Matches", systemImage: "trash")
                }
            }
            .disabled(model.isCleaning)

            if let result = model.cleanupResult {
                Text(cleanupSummary(result))
                    .font(.headline)
                List(result.stalePartials + result.incompleteSnapshots) { item in
                    HStack {
                        Text(item.name ?? item.path)
                            .lineLimit(1)
                        Spacer()
                        Text(DisplayFormat.bytes(item.size))
                            .foregroundStyle(.secondary)
                    }
                }
            } else {
                EmptyStateView(
                    title: "No Cleanup Scan",
                    systemImage: "trash",
                    detail: "Run a scan before deleting stale partial files."
                )
            }
            Spacer()
        }
        .padding(24)
        .navigationTitle("Cleanup")
    }

    private func cleanupSummary(_ result: CleanupResult) -> String {
        let action = result.dryRun == false ? "Deleted" : "Scan found"
        return "\(action) \(result.stalePartials.count) stale files and \(result.incompleteSnapshots.count) incomplete snapshots."
    }
}

struct SettingsView: View {
    private enum SettingsField: Hashable {
        case backendURL
        case pythonExecutable
    }

    @EnvironmentObject private var model: AppViewModel
    @FocusState private var focusedField: SettingsField?

    var body: some View {
        Form {
            Toggle("Launch local backend automatically", isOn: $model.autoLaunchBackend)
            FocusedTextField(
                placeholder: "Backend URL",
                text: $model.backendURL,
                focusedFieldValue: focusBinding,
                field: .backendURL
            )
                .focused($focusedField, equals: .backendURL)
            FocusedTextField(
                placeholder: "Python executable",
                text: $model.pythonExecutable,
                focusedFieldValue: focusBinding,
                field: .pythonExecutable
            )
                .focused($focusedField, equals: .pythonExecutable)
            HStack {
                Button("Launch Backend") { model.launchBackend() }
                Button("Stop Backend") { model.stopBackend() }
            }
        }
        .padding(24)
        .frame(width: 520)
        .onAppear {
            focusDefaultField()
        }
    }

    private func focusDefaultField() {
        Task { @MainActor in
            try? await Task.sleep(nanoseconds: 150_000_000)
            if focusedField == nil {
                focusedField = .backendURL
            }
        }
    }

    private var focusBinding: Binding<SettingsField?> {
        Binding(
            get: { focusedField },
            set: { focusedField = $0 }
        )
    }
}

struct StatusBadge: View {
    let status: QueueStatus

    var body: some View {
        Text(status.rawValue)
            .font(.caption2.weight(.semibold))
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(colour.opacity(0.15), in: Capsule())
            .foregroundStyle(colour)
    }

    private var colour: Color {
        switch status {
        case .waiting: .secondary
        case .running: .blue
        case .completed: .green
        case .failed: .red
        case .stopped: .orange
        case .unknown: .secondary
        }
    }
}

struct SectionHeader: View {
    let title: String
    let detail: String

    var body: some View {
        HStack(alignment: .firstTextBaseline) {
            Text(title)
                .font(.title3.weight(.semibold))
            Spacer()
            Text(detail)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }
}

struct NoticeBanner: View {
    let notice: Notice

    var body: some View {
        Text(notice.message)
            .font(.caption)
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .background(.regularMaterial, in: Capsule())
            .overlay {
                Capsule().stroke(notice.isError ? .red.opacity(0.45) : .green.opacity(0.45))
            }
    }
}

struct EmptyStateView: View {
    let title: String
    let systemImage: String
    let detail: String

    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: systemImage)
                .font(.system(size: 38))
                .foregroundStyle(.secondary)
            Text(title)
                .font(.headline)
            Text(detail)
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }
}
