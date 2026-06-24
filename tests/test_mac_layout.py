from pathlib import Path


VIEWS = Path("mac/HuggingFacePullMac/Sources/HuggingFacePullMac/Views.swift")


def test_search_layout_does_not_reserve_queue_detail_column():
    source = VIEWS.read_text()
    content_view = source[source.index("struct ContentView: View"):source.index("struct BackendStatusView: View")]

    assert "} content: {" not in content_view
    assert "QueueDetailView(item: model.selectedQueueItem)" not in content_view


def test_queue_screen_owns_queue_detail_column():
    source = VIEWS.read_text()
    queue_view = source[source.index("struct QueueView: View"):source.index("struct QueueRow: View")]

    assert "QueueDetailView(item: model.selectedQueueItem)" in queue_view


def test_search_view_has_explicit_keyboard_focus_targets():
    source = VIEWS.read_text()
    search_view = source[source.index("struct SearchView: View"):source.index("struct SearchResultRow: View")]

    assert "@FocusState private var focusedField: SearchField?" in search_view
    assert ".focused($focusedField, equals: .query)" in search_view
    assert ".focused($focusedField, equals: .repoID)" in search_view
    assert "FocusedTextField(" in search_view
    assert "focusedField = .query" in search_view
    assert "focusedField = .repoID" in search_view
    assert "Task.sleep" in search_view


def test_settings_view_focuses_backend_url_field():
    source = VIEWS.read_text()
    settings_view = source[source.index("struct SettingsView: View"):source.index("struct StatusBadge: View")]

    assert "@FocusState private var focusedField: SettingsField?" in settings_view
    assert ".focused($focusedField, equals: .backendURL)" in settings_view
    assert "FocusedTextField(" in settings_view
    assert "focusedField = .backendURL" in settings_view


def test_focused_text_field_uses_appkit_first_responder():
    source = VIEWS.read_text()
    helper = source[source.index("struct FocusedTextField<Field: Hashable>: NSViewRepresentable"):source.index("struct BackendStatusView: View")]

    assert "NSTextField" in helper
    assert "lastFocusedField" in helper
    assert "NSApplication.shared.activate" not in helper
    assert "window.makeKeyAndOrderFront(nil)" not in helper
    assert "window.makeFirstResponder(textField)" in helper


def test_app_shutdown_stops_backend_process():
    app = Path("mac/HuggingFacePullMac/Sources/HuggingFacePullMac/HuggingFacePullMacApp.swift").read_text()
    view_model = Path("mac/HuggingFacePullMac/Sources/HuggingFacePullMac/AppViewModel.swift").read_text()

    assert "@NSApplicationDelegateAdaptor(AppDelegate.self)" in app
    assert "applicationDidFinishLaunching" in app
    assert "NSApplication.shared.setActivationPolicy(.regular)" in app
    assert "NSApplication.shared.activate" in app
    assert "applicationWillTerminate" in app
    assert "applicationShouldTerminateAfterLastWindowClosed" in app
    assert "return true" in app
    assert "model.shutdown()" in app
    assert "func shutdown()" in view_model
    assert "pollTask?.cancel()" in view_model
    assert "backend.stop()" in view_model
