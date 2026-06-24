import Foundation

public extension JSONDecoder {
    static var hfp: JSONDecoder {
        JSONDecoder()
    }
}

public extension JSONEncoder {
    static var hfp: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return encoder
    }
}

public struct AppState: Decodable, Equatable, Sendable {
    public let running: Bool
    public let pauseRequested: Bool
    public let stopAfterFileRequested: Bool
    public let libraryDirectory: String
    public let endpoint: String
    public let installedModels: [InstalledModel]
    public let cachedModels: [InstalledModel]
    public let items: [QueueItem]

    enum CodingKeys: String, CodingKey {
        case running
        case pauseRequested = "pause_requested"
        case stopAfterFileRequested = "stop_after_file_requested"
        case libraryDirectory = "library_dir"
        case endpoint
        case installedModels = "installed_models"
        case cachedModels = "cached_models"
        case items
    }
}

public struct InstalledModel: Decodable, Identifiable, Equatable, Sendable {
    public var id: String {
        "\(repoID)@\(revision)#\(repoType)"
    }

    public let repoID: String
    public let revision: String
    public let repoType: RepoType
    public let size: Int64?
    public let snapshotPath: String?

    public var formattedSize: String {
        ByteFormat.string(fromByteCount: size ?? 0)
    }

    enum CodingKeys: String, CodingKey {
        case repoID = "repo_id"
        case revision
        case repoType = "repo_type"
        case size
        case snapshotPath = "snapshot_path"
    }
}

public struct QueueItem: Decodable, Identifiable, Equatable, Sendable {
    public let id: String
    public let repoID: String
    public let revision: String
    public let repoType: RepoType
    public let allowPatterns: [String]
    public let ignorePatterns: [String]
    public let deduplicated: Bool
    public let status: QueueStatus
    public let error: String?
    public let messages: [QueueMessage]
    public let progress: QueueProgress
    public let createdAt: TimeInterval
    public let updatedAt: TimeInterval

    public var progressFraction: Double? {
        guard let percent = progress.overall.percent else {
            return nil
        }
        return max(0, min(percent / 100, 1))
    }

    enum CodingKeys: String, CodingKey {
        case id
        case repoID = "repo_id"
        case revision
        case repoType = "repo_type"
        case allowPatterns = "allow_patterns"
        case ignorePatterns = "ignore_patterns"
        case deduplicated
        case status
        case error
        case messages
        case progress
        case createdAt = "created_at"
        case updatedAt = "updated_at"
    }
}

public enum QueueStatus: String, Decodable, Equatable, Sendable {
    case waiting
    case running
    case completed
    case failed
    case stopped
    case unknown

    public init(from decoder: Decoder) throws {
        let value = try decoder.singleValueContainer().decode(String.self)
        self = QueueStatus(rawValue: value) ?? .unknown
    }
}

public enum RepoType: String, Codable, CaseIterable, Equatable, Sendable {
    case model
    case dataset
    case space
}

public struct QueueMessage: Decodable, Equatable, Sendable {
    public let timestamp: TimeInterval?
    public let text: String?
}

public struct QueueProgress: Decodable, Equatable, Sendable {
    public let phase: String
    public let overall: OverallProgress
    public let currentFile: CurrentFileProgress?

    enum CodingKeys: String, CodingKey {
        case phase
        case overall
        case currentFile = "current_file"
    }
}

public struct OverallProgress: Decodable, Equatable, Sendable {
    public let downloaded: Int64
    public let total: Int64?
    public let percent: Double?
    public let bytesPerSecond: Double?
    public let etaSeconds: Double?

    enum CodingKeys: String, CodingKey {
        case downloaded
        case total
        case percent
        case bytesPerSecond = "bytes_per_second"
        case etaSeconds = "eta_seconds"
    }
}

public struct CurrentFileProgress: Decodable, Equatable, Sendable {
    public let path: String?
    public let name: String?
    public let digest: String?
    public let downloaded: Int64?
    public let total: Int64?
    public let index: Int?
    public let totalFiles: Int?
    public let updatedAt: TimeInterval?

    public var displayName: String {
        path ?? name ?? digest ?? "snapshot"
    }

    enum CodingKeys: String, CodingKey {
        case path
        case name
        case digest
        case downloaded
        case total
        case index
        case totalFiles = "total_files"
        case updatedAt = "updated_at"
    }
}

public struct SearchResponse: Decodable, Equatable, Sendable {
    public let available: Bool?
    public let results: [SearchResult]
    public let error: String?
}

public struct SearchResult: Decodable, Identifiable, Equatable, Sendable {
    public var id: String {
        repoID
    }

    public let repoID: String
    public let name: String?
    public let downloads: Int?
    public let likes: Int?
    public let pipelineTag: String?
    public let description: String?

    enum CodingKeys: String, CodingKey {
        case repoID = "repo_id"
        case name
        case downloads
        case likes
        case pipelineTag = "pipeline_tag"
        case description
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let decodedRepoID = try container.decodeIfPresent(String.self, forKey: .repoID)
        let decodedName = try container.decodeIfPresent(String.self, forKey: .name)
        repoID = decodedRepoID ?? decodedName ?? "unknown"
        name = decodedName
        downloads = try container.decodeIfPresent(Int.self, forKey: .downloads)
        likes = try container.decodeIfPresent(Int.self, forKey: .likes)
        pipelineTag = try container.decodeIfPresent(String.self, forKey: .pipelineTag)
        description = try container.decodeIfPresent(String.self, forKey: .description)
    }
}

public struct HubFileResponse: Decodable, Equatable, Sendable {
    public let files: [HubFile]
}

public struct HubFile: Decodable, Identifiable, Equatable, Sendable {
    public var id: String {
        path ?? name ?? UUID().uuidString
    }

    public let path: String?
    public let name: String?
    public let size: Int64?

    public var displayName: String {
        path ?? name ?? "unnamed file"
    }
}

public struct QueueRequest: Encodable, Equatable, Sendable {
    public let repoID: String
    public let revision: String
    public let repoType: RepoType
    public let allowPatterns: [String]
    public let ignorePatterns: [String]
    public let localDirectory: String?

    public init(
        repoID: String,
        revision: String = "main",
        repoType: RepoType = .model,
        allowPatterns: [String] = [],
        ignorePatterns: [String] = [],
        localDirectory: String? = nil
    ) {
        self.repoID = repoID
        self.revision = revision
        self.repoType = repoType
        self.allowPatterns = allowPatterns
        self.ignorePatterns = ignorePatterns
        self.localDirectory = localDirectory
    }

    enum CodingKeys: String, CodingKey {
        case repoID = "repo_id"
        case revision
        case repoType = "repo_type"
        case allowPatterns = "allow_patterns"
        case ignorePatterns = "ignore_patterns"
        case localDirectory = "local_dir"
    }
}

public struct InstalledRemoveRequest: Encodable, Equatable, Sendable {
    public let repoID: String
    public let revision: String
    public let repoType: RepoType

    public init(repoID: String, revision: String = "main", repoType: RepoType = .model) {
        self.repoID = repoID
        self.revision = revision
        self.repoType = repoType
    }

    enum CodingKeys: String, CodingKey {
        case repoID = "repo_id"
        case revision
        case repoType = "repo_type"
    }
}

public struct CleanupRequest: Encodable, Equatable, Sendable {
    public let includePartials: Bool
    public let olderThanDays: Int

    public init(includePartials: Bool, olderThanDays: Int) {
        self.includePartials = includePartials
        self.olderThanDays = olderThanDays
    }

    enum CodingKeys: String, CodingKey {
        case includePartials = "include_partials"
        case olderThanDays = "older_than_days"
    }
}

public struct CleanupResult: Decodable, Equatable, Sendable {
    public let dryRun: Bool?
    public let stalePartials: [CleanupItem]
    public let incompleteSnapshots: [CleanupItem]
    public let deleted: [String]
    public let deletedSnapshots: [String]

    enum CodingKeys: String, CodingKey {
        case dryRun = "dry_run"
        case stalePartials = "stale_partials"
        case incompleteSnapshots = "incomplete_snapshots"
        case deleted
        case deletedSnapshots = "deleted_snapshots"
    }
}

public struct CleanupItem: Decodable, Identifiable, Equatable, Sendable {
    public var id: String {
        path
    }

    public let path: String
    public let name: String?
    public let size: Int64?
    public let source: String?
}

public extension ByteCountFormatter {
    static var hfp: ByteCountFormatter {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        formatter.allowedUnits = [.useBytes, .useKB, .useMB, .useGB, .useTB]
        return formatter
    }
}

public enum ByteFormat {
    public static func string(fromByteCount bytes: Int64) -> String {
        let units = ["B", "KB", "MB", "GB", "TB"]
        var value = Double(bytes)
        var unitIndex = 0
        while value >= 1024, unitIndex < units.count - 1 {
            value /= 1024
            unitIndex += 1
        }
        if unitIndex == 0 {
            return "\(bytes) B"
        }
        return String(format: "%.2f %@", value, units[unitIndex])
    }
}
