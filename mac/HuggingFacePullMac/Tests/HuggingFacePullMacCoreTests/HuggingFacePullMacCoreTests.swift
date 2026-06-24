import Foundation
import Testing
@testable import HuggingFacePullMacCore

@Suite("HuggingFacePull Mac core", .serialized)
struct HuggingFacePullMacCoreTests {
    @Test("decodes queue state with progress and installed snapshots")
    func decodesQueueState() throws {
        let json = """
        {
          "running": true,
          "pause_requested": false,
          "stop_after_file_requested": false,
          "library_dir": "/Users/example/.cache/huggingfacepull/library",
          "endpoint": "https://huggingface.co",
          "installed_models": [
            {
              "repo_id": "Qwen/Qwen3",
              "revision": "main",
              "repo_type": "model",
              "size": 1048576,
              "snapshot_path": "/cache/models--Qwen--Qwen3/snapshots/abc"
            }
          ],
          "cached_models": [],
          "items": [
            {
              "id": "1",
              "repo_id": "Qwen/Qwen3",
              "revision": "main",
              "repo_type": "model",
              "allow_patterns": ["*.json"],
              "ignore_patterns": ["*.bin"],
              "deduplicated": false,
              "status": "running",
              "error": null,
              "messages": [{"timestamp": 1782014195, "text": "download-progress"}],
              "progress": {
                "phase": "downloading",
                "overall": {
                  "downloaded": 524288,
                  "total": 1048576,
                  "percent": 50,
                  "bytes_per_second": 131072,
                  "eta_seconds": 4
                },
                "current_file": {
                  "path": "model.safetensors",
                  "downloaded": 524288,
                  "total": 1048576,
                  "index": 2,
                  "total_files": 3,
                  "updated_at": 1782014195
                }
              },
              "created_at": 1782014000,
              "updated_at": 1782014195
            }
          ]
        }
        """.data(using: .utf8)!

        let state = try JSONDecoder.hfp.decode(AppState.self, from: json)

        #expect(state.running)
        #expect(state.libraryDirectory == "/Users/example/.cache/huggingfacepull/library")
        #expect(state.items.first?.repoID == "Qwen/Qwen3")
        #expect(state.items.first?.progress.overall.percent == 50)
        #expect(state.items.first?.progress.currentFile?.path == "model.safetensors")
        #expect(state.installedModels.first?.formattedSize == "1.00 MB")
    }

    @Test("API client sends expected requests and decodes responses")
    func apiClientRequests() async throws {
        MockURLProtocol.reset()
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [MockURLProtocol.self]
        let session = URLSession(configuration: configuration)
        let client = APIClient(
            baseURL: URL(string: "http://127.0.0.1:8019")!,
            session: session
        )

        MockURLProtocol.handler = { request in
            switch (request.httpMethod, request.url?.path, request.url?.query) {
            case ("GET", "/api/state", _):
                return .json("""
                {"running":false,"pause_requested":false,"stop_after_file_requested":false,"library_dir":"/tmp/lib","endpoint":"https://huggingface.co","installed_models":[],"cached_models":[],"items":[]}
                """)
            case ("GET", "/api/search", "q=qwen"):
                return .json("""
                {"available":true,"results":[{"repo_id":"Qwen/Qwen3","downloads":12,"likes":3,"pipeline_tag":"text-generation"}]}
                """)
            case ("POST", "/api/queue", _):
                let body = try #require(request.httpBodyStream?.readAllData())
                let text = try #require(String(data: body, encoding: .utf8))
                #expect(text.contains("\"repo_id\":\"Qwen/Qwen3\""))
                #expect(text.contains("\"allow_patterns\":[\"*.json\"]"))
                return .json("""
                {"id":"1","repo_id":"Qwen/Qwen3","revision":"main","repo_type":"model","allow_patterns":["*.json"],"ignore_patterns":[],"deduplicated":false,"status":"waiting","error":null,"messages":[],"progress":{"phase":"waiting","overall":{"downloaded":0,"total":null,"percent":null},"current_file":null},"created_at":1,"updated_at":1}
                """)
            case ("POST", "/api/start", _):
                return .json("""
                {"running":true,"pause_requested":false,"stop_after_file_requested":false,"library_dir":"/tmp/lib","endpoint":"https://huggingface.co","installed_models":[],"cached_models":[],"items":[]}
                """)
            case ("POST", "/api/cleanup/scan", _):
                return .json("""
                {"dry_run":true,"stale_partials":[{"path":"/tmp/partial","name":"partial","size":64,"source":"library"}],"incomplete_snapshots":[],"deleted":[],"deleted_snapshots":[]}
                """)
            default:
                return .response(statusCode: 404, body: Data())
            }
        }

        let state = try await client.fetchState()
        let search = try await client.search(query: "qwen")
        let queued = try await client.queue(
            QueueRequest(repoID: "Qwen/Qwen3", allowPatterns: ["*.json"])
        )
        let started = try await client.startQueue()
        let cleanup = try await client.scanCleanup(includePartials: true, olderThanDays: 0)

        #expect(state.libraryDirectory == "/tmp/lib")
        #expect(search.results.first?.repoID == "Qwen/Qwen3")
        #expect(queued.status == .waiting)
        #expect(started.running)
        #expect(cleanup.stalePartials.first?.size == 64)
    }

    @Test("API client surfaces server detail errors")
    func apiClientErrors() async throws {
        MockURLProtocol.reset()
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [MockURLProtocol.self]
        let client = APIClient(
            baseURL: URL(string: "http://127.0.0.1:8019")!,
            session: URLSession(configuration: configuration)
        )

        MockURLProtocol.handler = { _ in
            .response(statusCode: 409, body: #"{"detail":"Snapshot is already installed"}"#.data(using: .utf8)!)
        }

        do {
            _ = try await client.startQueue()
            Issue.record("Expected APIError.server to be thrown")
        } catch let error as APIError {
            #expect(error == .server("Snapshot is already installed"))
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }
}

private final class MockURLProtocol: URLProtocol, @unchecked Sendable {
    struct MockResponse: Sendable {
        let statusCode: Int
        let body: Data

        static func json(_ text: String) -> MockResponse {
            MockResponse(statusCode: 200, body: Data(text.utf8))
        }

        static func response(statusCode: Int, body: Data) -> MockResponse {
            MockResponse(statusCode: statusCode, body: body)
        }
    }

    static let lock = NSLock()
    nonisolated(unsafe) static var handler: (@Sendable (URLRequest) throws -> MockResponse)?

    static func reset() {
        lock.withLock {
            handler = nil
        }
    }

    override class func canInit(with request: URLRequest) -> Bool {
        true
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest {
        request
    }

    override func startLoading() {
        do {
            let handler = try Self.lock.withLock {
                try #require(Self.handler)
            }
            let response = try handler(request)
            let url = try #require(request.url)
            let http = try #require(HTTPURLResponse(
                url: url,
                statusCode: response.statusCode,
                httpVersion: nil,
                headerFields: ["Content-Type": "application/json"]
            ))
            client?.urlProtocol(self, didReceive: http, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: response.body)
            client?.urlProtocolDidFinishLoading(self)
        } catch {
            client?.urlProtocol(self, didFailWithError: error)
        }
    }

    override func stopLoading() {}
}

private extension InputStream {
    func readAllData() -> Data {
        open()
        defer { close() }
        var data = Data()
        let bufferSize = 4096
        let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: bufferSize)
        defer { buffer.deallocate() }

        while hasBytesAvailable {
            let count = read(buffer, maxLength: bufferSize)
            if count <= 0 {
                break
            }
            data.append(buffer, count: count)
        }
        return data
    }
}
