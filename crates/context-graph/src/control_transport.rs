use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::thread::JoinHandle;

/// Handler invoked by the live owner while it holds the underlying Store lock.
pub type ControlHandler = Arc<dyn Fn(String) -> String + Send + Sync + 'static>;

#[cfg(unix)]
mod unix {
    use std::fs;
    use std::io::{Read, Write};
    use std::os::unix::net::{UnixListener, UnixStream};
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread::{self, JoinHandle};
    use std::time::Duration;

    use super::ControlHandler;

    pub fn start_control_listener(
        socket_path: &Path,
        shutdown: Arc<AtomicBool>,
        handler: ControlHandler,
    ) -> Option<JoinHandle<()>> {
        if let Some(parent) = socket_path.parent()
            && let Err(error) = fs::create_dir_all(parent)
        {
            eprintln!(
                "owner transport directory setup failed path={} error={error}",
                parent.display()
            );
            return None;
        }
        let _ = fs::remove_file(socket_path);
        let listener = match UnixListener::bind(socket_path) {
            Ok(listener) => listener,
            Err(error) => {
                eprintln!(
                    "owner transport bind failed path={} error={error}",
                    socket_path.display()
                );
                return None;
            }
        };
        if let Err(error) = listener.set_nonblocking(true) {
            eprintln!("owner transport nonblocking setup failed error={error}");
            return None;
        }
        let socket_path = socket_path.to_path_buf();
        println!("owner transport listening path={}", socket_path.display());
        Some(thread::spawn(move || {
            accept_control_loop(listener, socket_path, shutdown, handler);
        }))
    }

    fn accept_control_loop(
        listener: UnixListener,
        socket_path: PathBuf,
        shutdown: Arc<AtomicBool>,
        handler: ControlHandler,
    ) {
        while !shutdown.load(Ordering::SeqCst) {
            match listener.accept() {
                Ok((mut stream, _addr)) => {
                    let mut frame = String::new();
                    let _ = stream.read_to_string(&mut frame);
                    let response = handler(frame);
                    let _ = stream.write_all(response.as_bytes());
                    let _ = stream.flush();
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(20));
                }
                Err(error) => {
                    eprintln!("owner transport accept failed error={error}");
                    break;
                }
            }
        }
        let _ = fs::remove_file(&socket_path);
    }

    pub fn request_control(socket_path: &Path, frame: &str) -> Result<String, String> {
        let mut stream = UnixStream::connect(socket_path)
            .map_err(|error| format!("could not connect to {}: {error}", socket_path.display()))?;
        stream
            .write_all(frame.as_bytes())
            .map_err(|error| format!("could not send owner request: {error}"))?;
        stream
            .shutdown(std::net::Shutdown::Write)
            .map_err(|error| format!("could not finish owner request: {error}"))?;
        let mut response = String::new();
        stream
            .read_to_string(&mut response)
            .map_err(|error| format!("could not read owner response: {error}"))?;
        Ok(response)
    }
}

#[cfg(unix)]
pub fn start_control_listener(
    socket_path: &Path,
    shutdown: Arc<AtomicBool>,
    handler: ControlHandler,
) -> Option<JoinHandle<()>> {
    unix::start_control_listener(socket_path, shutdown, handler)
}

#[cfg(not(unix))]
pub fn start_control_listener(
    _socket_path: &Path,
    _shutdown: Arc<AtomicBool>,
    _handler: ControlHandler,
) -> Option<JoinHandle<()>> {
    None
}

#[cfg(unix)]
pub fn request_control(socket_path: &Path, frame: &str) -> Result<String, String> {
    unix::request_control(socket_path, frame)
}

#[cfg(not(unix))]
pub fn request_control(_socket_path: &Path, _frame: &str) -> Result<String, String> {
    Err("owner control transport is only available on Unix".to_string())
}
