//! How much working memory one machine lets a cg write use, and where that number came from.
//!
//! ## The promise
//!
//! A write that does not fit is REFUSED, in words, with nothing half-written behind it — never the
//! kernel killing cg mid-import. The user sets nothing to get that: cg reads what the machine will
//! actually allow and sizes itself, and anyone who wants a different number says so with a knob.
//!
//! ## What is derived, and why these numbers
//!
//! The basis is the smaller of this machine's installed memory and the ceiling its control group
//! enforces, because a container on a large host lives under the container's number and sizing off
//! the host's is exactly how a container gets killed instead of refused.
//!
//! The share is a QUARTER of that basis. cg is not alone on the machine and not alone in its own
//! process: an embedding worker holds model weights, every read carries its own budget, and the
//! kernel wants page cache for the store file. A quarter leaves all three room while staying far
//! above what any single append, import batch or proof rebuild legitimately needs.
//!
//! The quarter is only ever capped, NEVER raised. A budget above what the kernel will actually
//! give this process is not a budget at all: the machine kills cg part-way through the write
//! instead of cg refusing it, which is the one outcome this whole mechanism exists to prevent. So
//! above [`WRITER_MEMORY_CEILING_BYTES`] the share is capped — nothing on cg's write path
//! legitimately allocates that much, and a larger number only postpones the refusal it exists to
//! give — and below [`WRITER_MEMORY_FLOOR_BYTES`] the share stands exactly as the machine allows
//! it.
//!
//! [`WRITER_MEMORY_FLOOR_BYTES`] is therefore a THRESHOLD, not a lower clamp: it is the point below
//! which a capture cannot assemble its proof chunks at all, so cg says so up front — naming the
//! share this machine allows and the knob that raises it — rather than opening a store on a budget
//! that will refuse every write.
//!
//! ## Who wins
//!
//! A switch on the command beats `[write] memory_limit_bytes` in `cg/fabric.toml`, which beats what
//! this module derives. There is no environment variable: a budget that changes with an exported
//! shell variable is a budget nobody can read back off the machine, and `cg status` prints both the
//! value in force and which of the three it came from.

use std::num::NonZeroUsize;

pub use cg_node_fs::MachineMemory;

/// The share of the machine one cg writer takes when nobody says otherwise: a quarter.
pub const WRITER_MEMORY_SHARE_DENOMINATOR: u64 = 4;

/// What a capture needs to assemble its proof chunks. A machine whose share lands under this is
/// told so before anything is opened — the share is never raised to meet it, because a budget above
/// what the kernel allows buys a kill instead of a refusal.
pub const WRITER_MEMORY_FLOOR_BYTES: usize = 256 * 1024 * 1024;

/// The largest derived writer budget. Past this the number stops guarding anything cg's write path
/// does; a bigger one only postpones the refusal.
pub const WRITER_MEMORY_CEILING_BYTES: usize = 8 * 1024 * 1024 * 1024;

/// Which of the three answers is in force, in the order they beat each other.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WriterMemoryLimitSource {
    /// Named on the command itself, for this invocation only.
    Switch,
    /// `[write] memory_limit_bytes` in this machine's `cg/fabric.toml`.
    ConfigFile,
    /// Nobody said anything, so cg read the machine.
    DerivedFromMachine {
        machine: MachineMemory,
        basis: MachineMemoryBasis,
    },
}

/// Which number the derivation was a share OF.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MachineMemoryBasis {
    /// The control group's ceiling — a container, or a systemd scope, holds this process under it.
    ContainerCeiling(u64),
    /// This machine's installed memory, with no tighter control-group ceiling in force.
    InstalledMemory(u64),
    /// Neither is published here, so the floor stands in for a number cg could not read.
    NeitherPublished,
}

/// The writer budget in force, and where it came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriterMemoryLimit {
    pub bytes: NonZeroUsize,
    pub source: WriterMemoryLimitSource,
}

impl WriterMemoryLimit {
    /// The budget the machine itself dictates, with nothing declared anywhere.
    pub fn derived_from(machine: MachineMemory) -> Self {
        let basis = match (machine.cgroup_limit_bytes, machine.total_bytes) {
            (Some(cgroup), Some(total)) if cgroup <= total => {
                MachineMemoryBasis::ContainerCeiling(cgroup)
            }
            (Some(cgroup), None) => MachineMemoryBasis::ContainerCeiling(cgroup),
            (_, Some(total)) => MachineMemoryBasis::InstalledMemory(total),
            (None, None) => MachineMemoryBasis::NeitherPublished,
        };
        let share = match basis {
            MachineMemoryBasis::ContainerCeiling(bytes)
            | MachineMemoryBasis::InstalledMemory(bytes) => {
                usize::try_from(bytes / WRITER_MEMORY_SHARE_DENOMINATOR)
                    .unwrap_or(WRITER_MEMORY_CEILING_BYTES)
            }
            // Nothing detected means nothing to exceed, so the starting budget is what a capture
            // needs; a write past it is refused in words like any other.
            MachineMemoryBasis::NeitherPublished => WRITER_MEMORY_FLOOR_BYTES,
        };
        // Capped, never raised. Handing out more than the machine allows is how a container gets
        // killed instead of refused, so the share is only ever brought DOWN to the cap. A share
        // that rounds to nothing becomes the smallest budget there is rather than being lifted to
        // meet anything -- it is far below what a capture needs and says so.
        let held = share.min(WRITER_MEMORY_CEILING_BYTES);
        Self {
            bytes: NonZeroUsize::new(held).unwrap_or(NonZeroUsize::MIN),
            source: WriterMemoryLimitSource::DerivedFromMachine { machine, basis },
        }
    }

    /// Whether this machine allows a cg writer less than one capture needs.
    ///
    /// Only a DERIVED budget answers true: a number the operator named with a knob is their call,
    /// and cg runs under it and refuses the writes that do not fit.
    pub fn is_below_what_a_write_needs(&self) -> bool {
        matches!(
            self.source,
            WriterMemoryLimitSource::DerivedFromMachine { .. }
        ) && self.bytes.get() < WRITER_MEMORY_FLOOR_BYTES
    }

    /// What to tell the operator when this machine allows less than a capture needs: the share in
    /// force, where it came from, what a capture needs, and the knob that raises it. `None` when
    /// the budget is workable.
    pub fn too_small_refusal(&self) -> Option<String> {
        self.is_below_what_a_write_needs().then(|| {
            format!(
                "this machine allows cg {} of working memory ({}), and a capture needs at least \
                 {} to assemble one capture batch, so nothing was opened; {}",
                human_bytes(self.bytes.get() as u64),
                self.origin(),
                human_bytes(WRITER_MEMORY_FLOOR_BYTES as u64),
                how_to_raise_the_writer_budget()
            )
        })
    }

    /// The budget this machine dictates right now.
    pub fn derived() -> Self {
        Self::derived_from(cg_node_fs::machine_memory())
    }

    /// The budget a switch on the command named.
    pub fn from_switch(bytes: NonZeroUsize) -> Self {
        Self {
            bytes,
            source: WriterMemoryLimitSource::Switch,
        }
    }

    /// The budget `cg/fabric.toml` named.
    pub fn from_config_file(bytes: NonZeroUsize) -> Self {
        Self {
            bytes,
            source: WriterMemoryLimitSource::ConfigFile,
        }
    }

    /// Where this number came from, in the words `cg status` prints.
    pub fn origin(&self) -> String {
        match &self.source {
            WriterMemoryLimitSource::Switch => {
                format!("set on this command by {WRITE_MEMORY_LIMIT_SWITCH}")
            }
            WriterMemoryLimitSource::ConfigFile => {
                format!(
                    "set in {WRITE_MEMORY_LIMIT_CONFIG_PATH} by {WRITE_MEMORY_LIMIT_CONFIG_KEY}"
                )
            }
            WriterMemoryLimitSource::DerivedFromMachine { basis, .. } => {
                let clamped = self.clamp_note();
                match basis {
                    MachineMemoryBasis::ContainerCeiling(bytes) => format!(
                        "derived from this machine: a quarter of the {} this process is allowed{clamped}",
                        human_bytes(*bytes)
                    ),
                    MachineMemoryBasis::InstalledMemory(bytes) => format!(
                        "derived from this machine: a quarter of the {} installed{clamped}",
                        human_bytes(*bytes)
                    ),
                    MachineMemoryBasis::NeitherPublished => {
                        "derived from this machine: what a capture needs, because this machine \
                         publishes neither its installed memory nor a ceiling on this process"
                            .to_string()
                    }
                }
            }
        }
    }

    fn clamp_note(&self) -> &'static str {
        let WriterMemoryLimitSource::DerivedFromMachine { basis, .. } = &self.source else {
            return "";
        };
        let unclamped = match basis {
            MachineMemoryBasis::ContainerCeiling(bytes)
            | MachineMemoryBasis::InstalledMemory(bytes) => bytes / WRITER_MEMORY_SHARE_DENOMINATOR,
            MachineMemoryBasis::NeitherPublished => return "",
        };
        if unclamped > WRITER_MEMORY_CEILING_BYTES as u64 {
            ", capped"
        } else {
            ""
        }
    }
}

/// The switch that names a writer budget for one command.
pub const WRITE_MEMORY_LIMIT_SWITCH: &str = "--write-memory-limit-bytes";

/// The config file that names one for every command on this machine.
pub const WRITE_MEMORY_LIMIT_CONFIG_PATH: &str = "cg/fabric.toml";

/// The key inside it.
pub const WRITE_MEMORY_LIMIT_CONFIG_KEY: &str = "[write] memory_limit_bytes";

/// Both knobs, in the words a refusal offers them.
pub fn how_to_raise_the_writer_budget() -> String {
    format!(
        "raise it with {WRITE_MEMORY_LIMIT_SWITCH} <BYTES> for this one command, or with \
         {WRITE_MEMORY_LIMIT_CONFIG_KEY} in {WRITE_MEMORY_LIMIT_CONFIG_PATH} for every command on \
         this machine"
    )
}

/// A byte count as a person reads it.
pub fn human_bytes(bytes: u64) -> String {
    const UNITS: [(&str, u64); 4] = [
        ("GiB", 1 << 30),
        ("MiB", 1 << 20),
        ("KiB", 1 << 10),
        ("bytes", 1),
    ];
    for (unit, size) in UNITS {
        if bytes >= size {
            if size == 1 {
                return format!("{bytes} bytes");
            }
            return format!("{:.1} {unit}", bytes as f64 / size as f64);
        }
    }
    format!("{bytes} bytes")
}

/// The writer budget a [`crate::StoreConfig`] gets when its caller names none: what this machine
/// dictates, read fresh each time so a config carried to another machine sizes itself there.
pub fn default_write_memory_limit_bytes() -> NonZeroUsize {
    WriterMemoryLimit::derived().bytes
}

#[cfg(test)]
mod what_this_machine_lets_one_write_hold {
    use super::*;

    const GIB: u64 = 1 << 30;
    const MIB: u64 = 1 << 20;

    fn machine(total_gib: Option<u64>, cgroup_gib: Option<u64>) -> MachineMemory {
        MachineMemory {
            total_bytes: total_gib.map(|value| value * GIB),
            cgroup_limit_bytes: cgroup_gib.map(|value| value * GIB),
        }
    }

    #[test]
    fn a_container_on_a_large_host_is_sized_from_the_container_not_the_host() {
        let limit = WriterMemoryLimit::derived_from(machine(Some(128), Some(2)));
        assert_eq!(2 * GIB / 4, limit.bytes.get() as u64);
        assert_eq!(
            "derived from this machine: a quarter of the 2.0 GiB this process is allowed",
            limit.origin(),
            "sizing off the host's memory is how a container gets killed instead of refused, so \
             the number the reader sees has to be the container's"
        );
    }

    #[test]
    fn a_plain_machine_with_no_ceiling_on_it_is_sized_from_what_is_installed() {
        let limit = WriterMemoryLimit::derived_from(machine(Some(16), None));
        assert_eq!(16 * GIB / 4, limit.bytes.get() as u64);
        assert!(
            limit
                .origin()
                .contains("a quarter of the 16.0 GiB installed"),
            "{}",
            limit.origin()
        );
    }

    /// The whole mechanism exists so a write that cannot fit is REFUSED in words instead of the
    /// kernel killing cg part-way through it. A budget above what the kernel will actually hand
    /// this process buys exactly the kill it was meant to prevent, so the share is capped and
    /// never raised — including when the share is smaller than a capture needs.
    #[test]
    fn a_budget_never_exceeds_the_memory_this_machine_will_actually_give_the_process() {
        for (total, ceiling) in [
            (128 * GIB, 128 * MIB),
            (64 * GIB, 512 * MIB),
            (8 * GIB, 64 * MIB),
            (512 * MIB, 512 * MIB),
        ] {
            let machine = MachineMemory {
                total_bytes: Some(total),
                cgroup_limit_bytes: Some(ceiling),
            };
            let limit = WriterMemoryLimit::derived_from(machine);
            assert_eq!(
                ceiling / WRITER_MEMORY_SHARE_DENOMINATOR,
                limit.bytes.get() as u64,
                "a {} container must get its own quarter, never a number the kernel would kill \
                 the process for reaching",
                human_bytes(ceiling)
            );
            assert!(
                (limit.bytes.get() as u64) < ceiling,
                "the budget must stay under the ceiling the kernel enforces"
            );
        }
    }

    /// The floor is a threshold, not a lower clamp: below it cg says so before opening anything,
    /// naming the share this machine allows and the knob that raises it.
    #[test]
    fn a_machine_that_allows_less_than_a_capture_needs_is_told_so_before_anything_is_opened() {
        let cramped = WriterMemoryLimit::derived_from(MachineMemory {
            total_bytes: Some(128 * GIB),
            cgroup_limit_bytes: Some(128 * MIB),
        });
        assert!(cramped.is_below_what_a_write_needs());
        let refusal = cramped
            .too_small_refusal()
            .expect("a machine that allows too little says so");
        assert!(refusal.contains("32.0 MiB"), "{refusal}");
        assert!(
            refusal.contains("128.0 MiB this process is allowed"),
            "{refusal}"
        );
        assert!(refusal.contains("256.0 MiB"), "{refusal}");
        assert!(
            refusal.contains("to assemble one capture batch"),
            "{refusal}"
        );
        assert!(refusal.contains(WRITE_MEMORY_LIMIT_SWITCH), "{refusal}");
        assert!(refusal.contains(WRITE_MEMORY_LIMIT_CONFIG_KEY), "{refusal}");
    }

    /// Exactly at the threshold, and one step either side of it.
    #[test]
    fn the_threshold_is_the_share_a_capture_needs_and_nothing_below_it_is_raised_to_meet_it() {
        let at = MachineMemory {
            total_bytes: Some(128 * GIB),
            cgroup_limit_bytes: Some(
                WRITER_MEMORY_FLOOR_BYTES as u64 * WRITER_MEMORY_SHARE_DENOMINATOR,
            ),
        };
        let below = MachineMemory {
            cgroup_limit_bytes: at
                .cgroup_limit_bytes
                .map(|bytes| bytes - WRITER_MEMORY_SHARE_DENOMINATOR),
            ..at
        };
        let above = MachineMemory {
            cgroup_limit_bytes: at
                .cgroup_limit_bytes
                .map(|bytes| bytes + WRITER_MEMORY_SHARE_DENOMINATOR),
            ..at
        };

        let at = WriterMemoryLimit::derived_from(at);
        assert_eq!(WRITER_MEMORY_FLOOR_BYTES, at.bytes.get());
        assert!(!at.is_below_what_a_write_needs());
        assert_eq!(None, at.too_small_refusal());

        let below = WriterMemoryLimit::derived_from(below);
        assert_eq!(WRITER_MEMORY_FLOOR_BYTES - 1, below.bytes.get());
        assert!(below.is_below_what_a_write_needs());

        let above = WriterMemoryLimit::derived_from(above);
        assert_eq!(WRITER_MEMORY_FLOOR_BYTES + 1, above.bytes.get());
        assert!(!above.is_below_what_a_write_needs());
    }

    /// A number the operator named is their call, however small; cg runs under it and refuses the
    /// writes that do not fit.
    #[test]
    fn a_budget_an_operator_named_is_never_second_guessed_by_the_threshold() {
        let tiny = NonZeroUsize::new(4096).expect("non-zero");
        assert!(!WriterMemoryLimit::from_switch(tiny).is_below_what_a_write_needs());
        assert!(!WriterMemoryLimit::from_config_file(tiny).is_below_what_a_write_needs());
        assert_eq!(
            None,
            WriterMemoryLimit::from_switch(tiny).too_small_refusal()
        );
    }

    #[test]
    fn a_very_large_machine_is_capped_because_a_bigger_number_guards_nothing() {
        let limit = WriterMemoryLimit::derived_from(machine(Some(512), None));
        assert_eq!(WRITER_MEMORY_CEILING_BYTES, limit.bytes.get());
        assert!(limit.origin().ends_with(", capped"), "{}", limit.origin());
    }

    #[test]
    fn a_machine_that_publishes_neither_number_still_gets_a_workable_budget() {
        let limit = WriterMemoryLimit::derived_from(machine(None, None));
        assert_eq!(WRITER_MEMORY_FLOOR_BYTES, limit.bytes.get());
        assert!(
            limit.origin().contains("publishes neither"),
            "cg never fails for want of a number it could not read; it says which number it could \
             not read: {}",
            limit.origin()
        );
    }

    #[test]
    fn a_declared_budget_says_which_knob_declared_it() {
        let bytes = NonZeroUsize::new(7 * MIB as usize).expect("non-zero");
        assert!(
            WriterMemoryLimit::from_switch(bytes)
                .origin()
                .contains(WRITE_MEMORY_LIMIT_SWITCH)
        );
        assert!(
            WriterMemoryLimit::from_config_file(bytes)
                .origin()
                .contains(WRITE_MEMORY_LIMIT_CONFIG_KEY)
        );
    }

    #[test]
    fn a_refusal_offers_both_knobs_and_no_environment_variable() {
        let offered = how_to_raise_the_writer_budget();
        assert!(offered.contains(WRITE_MEMORY_LIMIT_SWITCH), "{offered}");
        assert!(offered.contains(WRITE_MEMORY_LIMIT_CONFIG_KEY), "{offered}");
        assert!(
            !offered.contains("CG_") && !offered.to_lowercase().contains("environment"),
            "a budget that moves with an exported shell variable is one nobody can read back off \
             the machine: {offered}"
        );
    }
}

#[cfg(test)]
mod what_this_machine_actually_reports {
    use super::*;

    /// The derivation has to agree with what this machine really publishes, not only with the
    /// numbers a test hands it: a probe that silently reads nothing would leave every machine on
    /// the floor and nothing would say so.
    #[test]
    fn the_live_probe_reads_this_machine_and_the_origin_names_what_it_read() {
        let machine = cg_node_fs::machine_memory();
        let limit = WriterMemoryLimit::derived_from(machine);
        let origin = limit.origin();
        // Captured by default; `--nocapture` shows what this machine actually reported.
        println!(
            "{machine:?} -> {} ({origin})",
            human_bytes(limit.bytes.get() as u64)
        );

        match machine.cgroup_limit_bytes {
            // Under a container or a systemd scope the ceiling is the one that binds, and it is
            // never larger than what is installed.
            Some(ceiling) => {
                if let Some(total) = machine.total_bytes {
                    assert!(
                        ceiling <= total,
                        "a control-group ceiling above installed memory is not a ceiling: \
                         {ceiling} > {total}"
                    );
                }
                assert!(
                    origin.contains("this process is allowed"),
                    "a process the kernel holds under a ceiling is sized from that ceiling: \
                     {origin}"
                );
                assert_eq!(
                    (ceiling / WRITER_MEMORY_SHARE_DENOMINATOR).clamp(
                        WRITER_MEMORY_FLOOR_BYTES as u64,
                        WRITER_MEMORY_CEILING_BYTES as u64
                    ),
                    limit.bytes.get() as u64,
                    "the number in force is a quarter of the ceiling the kernel enforces, held \
                     inside the floor and the cap"
                );
            }
            None => {
                let total = machine
                    .total_bytes
                    .expect("this machine publishes its installed memory");
                assert!(total > 0);
                assert!(origin.contains("installed"), "{origin}");
            }
        }
        assert!(limit.bytes.get() <= WRITER_MEMORY_CEILING_BYTES);
        // Never more than this machine will actually hand the process.
        for detected in [machine.cgroup_limit_bytes, machine.total_bytes]
            .into_iter()
            .flatten()
        {
            assert!(
                (limit.bytes.get() as u64) <= detected / WRITER_MEMORY_SHARE_DENOMINATOR,
                "the derived budget must never exceed its detected share of {detected}"
            );
        }
    }
}
