import Foundation

public enum APIError: Error, Equatable, LocalizedError, Sendable {
    case invalidResponse
    case server(String)
    case transport(String)

    public var errorDescription: String? {
        switch self {
        case .invalidResponse:
            "The server returned an invalid response."
        case .server(let message):
            message
        case .transport(let message):
            message
        }
    }
}

public struct APIClient: Sendable {
    public var baseURL: URL
    private let session: URLSession
    private let decoder: JSONDecoder
    private let encoder: JSONEncoder

    public init(
        baseURL: URL,
        session: URLSession = .shared,
        decoder: JSONDecoder = .hfp,
        encoder: JSONEncoder = .hfp
    ) {
        self.baseURL = baseURL
        self.session = session
        self.decoder = decoder
        self.encoder = encoder
    }

    public func fetchState() async throws -> AppState {
        try await get("/api/state")
    }

    public func search(query: String) async throws -> SearchResponse {
        var components = URLComponents()
        components.path = "/api/search"
        components.queryItems = [URLQueryItem(name: "q", value: query)]
        return try await get(components)
    }

    public func files(repoID: String, revision: String = "main", repoType: RepoType = .model) async throws -> HubFileResponse {
        var components = URLComponents()
        components.path = "/api/models/\(repoPath(repoID))/files"
        components.queryItems = [
            URLQueryItem(name: "revision", value: revision),
            URLQueryItem(name: "repo_type", value: repoType.rawValue),
        ]
        return try await get(components)
    }

    public func queue(_ payload: QueueRequest) async throws -> QueueItem {
        try await post("/api/queue", body: payload)
    }

    public func startQueue() async throws -> AppState {
        try await post("/api/start", body: EmptyBody())
    }

    public func pauseQueue() async throws -> AppState {
        try await post("/api/pause", body: EmptyBody())
    }

    public func stopAfterCurrentFile() async throws -> AppState {
        try await post("/api/stop-after-file", body: EmptyBody())
    }

    public func retry(itemID: String) async throws -> QueueItem {
        try await post("/api/retry/\(pathComponent(itemID))", body: EmptyBody())
    }

    public func remove(itemID: String) async throws {
        let _: OKResponse = try await post("/api/remove/\(pathComponent(itemID))", body: EmptyBody())
    }

    public func removeInstalled(_ payload: InstalledRemoveRequest) async throws {
        let _: OKResponse = try await post("/api/installed/remove", body: payload)
    }

    public func scanCleanup(includePartials: Bool, olderThanDays: Int) async throws -> CleanupResult {
        try await post(
            "/api/cleanup/scan",
            body: CleanupRequest(includePartials: includePartials, olderThanDays: olderThanDays)
        )
    }

    public func deleteCleanup(includePartials: Bool, olderThanDays: Int) async throws -> CleanupResult {
        try await post(
            "/api/cleanup/delete",
            body: CleanupRequest(includePartials: includePartials, olderThanDays: olderThanDays)
        )
    }

    private func get<T: Decodable>(_ path: String) async throws -> T {
        var components = URLComponents()
        components.path = path
        return try await get(components)
    }

    private func get<T: Decodable>(_ components: URLComponents) async throws -> T {
        let url = try makeURL(components)
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        return try await send(request)
    }

    private func post<Body: Encodable, T: Decodable>(_ path: String, body: Body) async throws -> T {
        var components = URLComponents()
        components.path = path
        let url = try makeURL(components)
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        if Body.self != EmptyBody.self {
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = try encoder.encode(body)
        }
        return try await send(request)
    }

    private func send<T: Decodable>(_ request: URLRequest) async throws -> T {
        do {
            let (data, response) = try await session.data(for: request)
            guard let http = response as? HTTPURLResponse else {
                throw APIError.invalidResponse
            }
            guard (200..<300).contains(http.statusCode) else {
                throw decodeServerError(from: data) ?? APIError.server("HTTP \(http.statusCode)")
            }
            if T.self == EmptyResponse.self {
                return EmptyResponse() as! T
            }
            return try decoder.decode(T.self, from: data)
        } catch let error as APIError {
            throw error
        } catch {
            throw APIError.transport(error.localizedDescription)
        }
    }

    private func decodeServerError(from data: Data) -> APIError? {
        guard
            let decoded = try? decoder.decode(ServerError.self, from: data),
            let message = decoded.detail ?? decoded.error ?? decoded.message
        else {
            return nil
        }
        return .server(message)
    }

    private func makeURL(_ components: URLComponents) throws -> URL {
        guard let relativeURL = components.url(relativeTo: baseURL) else {
            throw APIError.invalidResponse
        }
        return relativeURL.absoluteURL
    }

    private func repoPath(_ repoID: String) -> String {
        repoID
            .split(separator: "/")
            .map { pathComponent(String($0)) }
            .joined(separator: "/")
    }

    private func pathComponent(_ value: String) -> String {
        value.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? value
    }
}

private struct EmptyBody: Encodable {}
private struct EmptyResponse: Decodable {}
private struct OKResponse: Decodable {
    let ok: Bool
}

private struct ServerError: Decodable {
    let detail: String?
    let error: String?
    let message: String?
}
