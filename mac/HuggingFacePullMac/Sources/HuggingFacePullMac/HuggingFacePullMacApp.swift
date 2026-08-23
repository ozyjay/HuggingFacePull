import AppKit
import SwiftUI
import HuggingFacePullMacCore

final class AppDelegate: NSObject, NSApplicationDelegate {
    var shutdown: (() -> Void)?

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApplication.shared.setActivationPolicy(.regular)
        NSApplication.shared.activate(ignoringOtherApps: true)
    }

    func applicationWillTerminate(_ notification: Notification) {
        shutdown?()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }
}

@main
struct HuggingFacePullMacApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var model = AppViewModel()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(model)
                .frame(minWidth: 1100, minHeight: 720)
                .onAppear {
                    appDelegate.shutdown = {
                        model.shutdown()
                    }
                }
                .onDisappear {
                    model.shutdown()
                }
                .task {
                    await model.start()
                }
        }
        Settings {
            SettingsView()
                .environmentObject(model)
        }
    }
}
