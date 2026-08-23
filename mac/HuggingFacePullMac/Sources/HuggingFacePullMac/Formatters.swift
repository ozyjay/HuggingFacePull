import Foundation
import HuggingFacePullMacCore

enum DisplayFormat {
    static func bytes(_ bytes: Int64?) -> String {
        ByteFormat.string(fromByteCount: bytes ?? 0)
    }

    static func percent(_ percent: Double?) -> String {
        guard let percent else {
            return "calculating"
        }
        return String(format: "%.1f%%", percent)
    }

    static func speed(_ bytesPerSecond: Double?) -> String {
        guard let bytesPerSecond, bytesPerSecond > 0 else {
            return "no speed"
        }
        return "\(ByteFormat.string(fromByteCount: Int64(bytesPerSecond)))/s"
    }

    static func duration(_ seconds: Double?) -> String {
        guard let seconds else {
            return "unknown"
        }
        let value = Int(seconds)
        if value < 60 {
            return "\(value)s"
        }
        return "\(value / 60)m \(value % 60)s"
    }

    static func timestamp(_ time: TimeInterval?) -> String {
        guard let time else {
            return "unknown"
        }
        return Date(timeIntervalSince1970: time).formatted(date: .omitted, time: .standard)
    }
}
