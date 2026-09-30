//! The working-memory ceiling this invocation gives the store it writes to.
//!
//! Three answers, in the order they beat each other: the switch on the command, then
//! `[write] memory_limit_bytes` in this machine's `cg/fabric.toml`, then what
//! [`WriterMemoryLimit::derived`] reads off the machine. No environment variable — a ceiling that
//! moves with an exported shell variable is one nobody can read back off the machine, and
//! `cg status` prints both the number in force and which of the three set it.
//!
//! Resolved once, before any store is opened, and held for the rest of the invocation so every
//! foreground store this command opens uses the ceiling the status output names. Detached search
//! batches use the ceilings persisted with their pending import requests.

use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::OnceLock;

use context_graph::write_budget::{
    WRITE_MEMORY_LIMIT_CONFIG_KEY, WRITE_MEMORY_LIMIT_SWITCH, WriterMemoryLimit,
};

static IN_FORCE: OnceLock<WriterMemoryLimit> = OnceLock::new();

/// Lift the switch out of the argument list before any command parses its own flags, so one
/// invocation-wide setting does not have to be admitted by every command's flag list.
///
/// Returns the remaining arguments and the value the switch named, if it was given.
pub(crate) fn take_switch(
    args: Vec<String>,
) -> Result<(Vec<String>, Option<NonZeroUsize>), String> {
    let mut remaining = Vec::with_capacity(args.len());
    let mut named = None;
    let mut arguments = args.into_iter();
    while let Some(argument) = arguments.next() {
        let value = if argument == WRITE_MEMORY_LIMIT_SWITCH {
            arguments.next().ok_or_else(|| {
                format!("{WRITE_MEMORY_LIMIT_SWITCH} needs a positive whole number of bytes")
            })?
        } else if let Some(inline) = argument.strip_prefix(&format!("{WRITE_MEMORY_LIMIT_SWITCH}="))
        {
            inline.to_string()
        } else {
            remaining.push(argument);
            continue;
        };
        named = Some(parse_bytes(&value).ok_or_else(|| {
            format!(
                "{WRITE_MEMORY_LIMIT_SWITCH} must be a positive whole number of bytes, found \
                 `{value}`"
            )
        })?);
    }
    Ok((remaining, named))
}

fn parse_bytes(value: &str) -> Option<NonZeroUsize> {
    value.parse::<usize>().ok().and_then(NonZeroUsize::new)
}

/// Which of the three answers this invocation runs under.
pub(crate) fn resolve(
    root: &Path,
    switch: Option<NonZeroUsize>,
) -> Result<WriterMemoryLimit, String> {
    if let Some(bytes) = switch {
        return Ok(WriterMemoryLimit::from_switch(bytes));
    }
    if let Some(bytes) = configured(root)? {
        return Ok(WriterMemoryLimit::from_config_file(bytes));
    }
    // Inspection still needs to report the actual share on a machine too small to capture.
    Ok(WriterMemoryLimit::derived())
}

/// Capture admission applies when a command enters writable work, after the shared setting has
/// been resolved for both read and write commands.
pub(crate) fn capture_admission_refusal(
    args: &[String],
    limit: &WriterMemoryLimit,
) -> Option<String> {
    let command = args.first().map(String::as_str)?;
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        return None;
    }
    let writable = match command {
        "record" | "understand" | "worker" | "fix" | "drop" | "group" | "undo" | "hook" => true,
        "import" => !args.iter().any(|arg| arg == "--dry-run"),
        "sync" => args.get(1).is_some_and(|mode| mode != "status"),
        _ => false,
    };
    writable.then(|| limit.too_small_refusal()).flatten()
}

fn configured(root: &Path) -> Result<Option<NonZeroUsize>, String> {
    let Some((path, config)) = super::history::cg_config(root)? else {
        return Ok(None);
    };
    let invalid = || {
        format!(
            "{}: {WRITE_MEMORY_LIMIT_CONFIG_KEY} must be a positive whole number of bytes",
            path.display()
        )
    };
    let Some(write) = config.get("write") else {
        return Ok(None);
    };
    let table = write.as_table().ok_or_else(invalid)?;
    let Some(value) = table.get("memory_limit_bytes") else {
        return Ok(None);
    };
    value
        .as_integer()
        .and_then(|bytes| usize::try_from(bytes).ok())
        .and_then(NonZeroUsize::new)
        .map(Some)
        .ok_or_else(invalid)
}

/// Hold the resolved ceiling for the rest of this invocation.
pub(crate) fn install(limit: WriterMemoryLimit) {
    let _ = IN_FORCE.set(limit);
}

/// The ceiling in force. An invocation that never resolved one — a library test, or a command that
/// refused before reaching the store — reads the machine, which is the same answer resolution
/// would have reached with nothing declared.
pub(crate) fn in_force() -> WriterMemoryLimit {
    IN_FORCE
        .get()
        .cloned()
        .unwrap_or_else(WriterMemoryLimit::derived)
}

#[cfg(test)]
mod which_answer_this_invocation_runs_under {
    use super::*;

    fn write_config(root: &Path, body: &str) {
        let directory = root.join("cg");
        std::fs::create_dir_all(&directory).expect("create an isolated config directory");
        std::fs::write(directory.join("fabric.toml"), body).expect("write the isolated config");
    }

    fn bytes(value: usize) -> NonZeroUsize {
        NonZeroUsize::new(value).expect("non-zero")
    }

    fn args(items: &[&str]) -> Vec<String> {
        items.iter().map(|item| (*item).to_string()).collect()
    }

    #[test]
    fn the_switch_is_lifted_off_the_argument_list_so_every_command_accepts_it() {
        let (remaining, named) =
            take_switch(args(&["import", ".", WRITE_MEMORY_LIMIT_SWITCH, "4096"]))
                .expect("the switch parses");
        assert_eq!(args(&["import", "."]), remaining);
        assert_eq!(Some(bytes(4096)), named);

        let (remaining, named) = take_switch(args(&[
            "status",
            &format!("{WRITE_MEMORY_LIMIT_SWITCH}=8192"),
        ]))
        .expect("the joined spelling parses too");
        assert_eq!(args(&["status"]), remaining);
        assert_eq!(Some(bytes(8192)), named);
    }

    #[test]
    fn a_switch_with_no_usable_number_is_refused_before_any_store_is_opened() {
        for bad in ["0", "-1", "lots", ""] {
            let refusal = take_switch(args(&["status", WRITE_MEMORY_LIMIT_SWITCH, bad]))
                .expect_err("a budget that is not a positive byte count is refused");
            assert!(refusal.contains(WRITE_MEMORY_LIMIT_SWITCH), "{refusal}");
        }
        let refusal = take_switch(args(&["status", WRITE_MEMORY_LIMIT_SWITCH]))
            .expect_err("a switch with nothing after it is refused");
        assert!(
            refusal.contains("needs a positive whole number"),
            "{refusal}"
        );
    }

    #[test]
    fn the_switch_beats_the_config_file_which_beats_what_the_machine_dictates() {
        let root = tempfile::tempdir().expect("isolated config root");
        write_config(root.path(), "[write]\nmemory_limit_bytes = 12345678\n");

        assert_eq!(
            WriterMemoryLimit::from_switch(bytes(999_000)),
            resolve(root.path(), Some(bytes(999_000))).expect("the switch resolves"),
            "a number named on the command is the one that command runs under"
        );
        assert_eq!(
            WriterMemoryLimit::from_config_file(bytes(12_345_678)),
            resolve(root.path(), None).expect("the config file resolves"),
            "with no switch, this machine's own config file decides"
        );

        let bare = tempfile::tempdir().expect("isolated config root");
        let derived = WriterMemoryLimit::derived();
        assert_eq!(
            derived,
            resolve(bare.path(), None).expect("the machine resolves"),
            "inspection reports the machine's actual share even below the capture threshold"
        );
    }

    #[test]
    fn a_small_container_allows_inspection_but_refuses_writable_capture() {
        use context_graph::write_budget::{MachineMemory, WriterMemoryLimit};

        let limit = WriterMemoryLimit::derived_from(MachineMemory {
            total_bytes: Some(16 * 1024 * 1024 * 1024),
            cgroup_limit_bytes: Some(512 * 1024 * 1024),
        });
        assert_eq!(128 * 1024 * 1024, limit.bytes.get());
        let refusal = limit.too_small_refusal().expect("capture is too large");
        for command in [
            args(&["--help"]),
            args(&["status", "--context", "memory"]),
            args(&["show", "00000000-0000-0000-0000-000000000000"]),
            args(&["find", "a question"]),
            args(&["import", ".", "--dry-run"]),
            args(&["forget", "--context", "memory"]),
        ] {
            assert_eq!(None, capture_admission_refusal(&command, &limit));
        }
        assert_eq!(
            Some(refusal),
            capture_admission_refusal(&args(&["import", "."]), &limit)
        );
    }

    /// A budget named with a knob is the operator's call, however small: resolution hands it back
    /// and cg runs under it. Only a DERIVED budget below what a capture needs stops the command.
    #[test]
    fn a_named_budget_below_what_a_capture_needs_is_still_the_number_that_runs() {
        let root = tempfile::tempdir().expect("isolated config root");
        let tiny = bytes(4096);
        assert_eq!(
            WriterMemoryLimit::from_switch(tiny),
            resolve(root.path(), Some(tiny)).expect("a named budget resolves whatever its size")
        );
        write_config(root.path(), "[write]\nmemory_limit_bytes = 4096\n");
        assert_eq!(
            WriterMemoryLimit::from_config_file(tiny),
            resolve(root.path(), None).expect("a configured budget resolves whatever its size")
        );
    }

    #[test]
    fn a_config_file_whose_budget_is_not_a_byte_count_is_named_with_its_path() {
        let root = tempfile::tempdir().expect("isolated config root");
        for body in [
            "[write]\nmemory_limit_bytes = 0\n",
            "[write]\nmemory_limit_bytes = \"lots\"\n",
            "[write]\nmemory_limit_bytes = -5\n",
            "write = 3\n",
        ] {
            write_config(root.path(), body);
            let refusal = resolve(root.path(), None)
                .expect_err("a budget that is not a positive byte count is refused");
            assert!(refusal.contains("fabric.toml"), "{refusal}");
            assert!(refusal.contains("memory_limit_bytes"), "{refusal}");
        }
    }
}
