use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use context_graph::{ControlHandler, request_control, start_control_listener};

#[cfg(unix)]
#[test]
fn owner_control_transport_is_public_handler_driven_and_replaces_stale_socket() {
    let temp = tempfile::tempdir().expect("temp dir");
    let socket_path = temp.path().join("owner").join("control.sock");
    std::fs::create_dir_all(socket_path.parent().expect("socket parent")).expect("socket parent");
    std::fs::write(&socket_path, b"stale socket placeholder").expect("stale socket placeholder");

    let shutdown = Arc::new(AtomicBool::new(false));
    let handler: ControlHandler = Arc::new(|frame| format!("handled:{}", frame.trim()));
    let listener = start_control_listener(&socket_path, shutdown.clone(), handler)
        .expect("listener starts on a path with a stale socket file");

    wait_for_socket(&socket_path);

    let response = request_control(&socket_path, "ping\n").expect("control round trip");
    assert_eq!(response, "handled:ping");

    shutdown.store(true, Ordering::SeqCst);
    listener.join().expect("listener exits cleanly");
    assert!(
        !socket_path.exists(),
        "control listener must remove its socket file on shutdown"
    );
}

#[cfg(not(unix))]
#[test]
fn owner_control_transport_reports_unavailable_on_non_unix() {
    let shutdown = Arc::new(AtomicBool::new(false));
    let handler: ControlHandler = Arc::new(|frame| frame);
    assert!(
        start_control_listener(std::path::Path::new("control.sock"), shutdown, handler).is_none()
    );
    assert!(
        request_control(std::path::Path::new("control.sock"), "ping").is_err(),
        "non-Unix platforms must get an explicit unavailable error"
    );
}

#[cfg(unix)]
fn wait_for_socket(path: &std::path::Path) {
    let started = Instant::now();
    while started.elapsed() < Duration::from_secs(2) {
        if path.exists() {
            return;
        }
        thread::sleep(Duration::from_millis(10));
    }
    panic!("control socket did not appear at {}", path.display());
}
