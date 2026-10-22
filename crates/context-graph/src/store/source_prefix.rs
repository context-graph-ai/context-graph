//! Bounded evidence that a checkpointed source still begins with the bytes cg committed.
//!
//! Resuming a source must seek past the completed prefix without reading it back, so an append
//! never re-hashes history. A checkpoint therefore stores, beside the SHA-256 of the exact committed
//! prefix, the hash state at the prefix's last 64-byte block boundary and the digests of two fixed
//! windows: the first [`WINDOW`] bytes of the source and the [`WINDOW`] bytes just before the
//! checkpoint. Resuming reads only those windows (at most `2 * WINDOW` bytes), checks them, and
//! continues the exact prefix hash from the stored state. A source that was truncated, replaced, or
//! rewritten at its head or near the checkpoint is refused; a full re-hash of the whole prefix stays
//! available on demand (`check_event_source_prefix_fully`), never on an append.

use std::collections::VecDeque;
use std::io::SeekFrom;

use sha2::digest::generic_array::GenericArray;
use sha2::{Digest as _, Sha256};

use crate::{CgError, Result, SeekableEventSource};

/// Bytes in each verification window.
pub(crate) const WINDOW: u64 = 64 * 1024;

const SHA256_INITIAL: [u32; 8] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];

/// SHA-256 whose state can be stored at a block boundary and resumed from it.
#[derive(Clone)]
pub(crate) struct ResumableSha256 {
    state: [u32; 8],
    pending: Vec<u8>,
    total: u64,
}

impl ResumableSha256 {
    pub(crate) fn new() -> Self {
        Self {
            state: SHA256_INITIAL,
            pending: Vec::with_capacity(64),
            total: 0,
        }
    }

    /// Resume from a stored block-boundary state: `boundary` bytes have been hashed.
    fn resume(state_hex: &str, boundary: u64) -> Option<Self> {
        if !boundary.is_multiple_of(64) || state_hex.len() != 64 {
            return None;
        }
        let mut state = [0_u32; 8];
        for (word, chunk) in state.iter_mut().zip(state_hex.as_bytes().chunks(8)) {
            *word = u32::from_str_radix(std::str::from_utf8(chunk).ok()?, 16).ok()?;
        }
        Some(Self {
            state,
            pending: Vec::with_capacity(64),
            total: boundary,
        })
    }

    pub(crate) fn update(&mut self, mut data: &[u8]) {
        self.total += data.len() as u64;
        if !self.pending.is_empty() {
            let wanted = (64 - self.pending.len()).min(data.len());
            self.pending.extend_from_slice(&data[..wanted]);
            data = &data[wanted..];
            if self.pending.len() < 64 {
                return;
            }
            let block = GenericArray::clone_from_slice(&self.pending);
            sha2::compress256(&mut self.state, &[block]);
            self.pending.clear();
        }
        let full = data.len() / 64 * 64;
        if full > 0 {
            let blocks = data[..full]
                .chunks_exact(64)
                .map(GenericArray::clone_from_slice)
                .collect::<Vec<_>>();
            sha2::compress256(&mut self.state, &blocks);
        }
        self.pending.extend_from_slice(&data[full..]);
    }

    /// Hex SHA-256 of everything hashed so far.
    pub(crate) fn digest_hex(&self) -> String {
        let mut state = self.state;
        let mut tail = self.pending.clone();
        tail.push(0x80);
        while tail.len() % 64 != 56 {
            tail.push(0);
        }
        tail.extend_from_slice(&(self.total * 8).to_be_bytes());
        let blocks = tail
            .chunks_exact(64)
            .map(GenericArray::clone_from_slice)
            .collect::<Vec<_>>();
        sha2::compress256(&mut state, &blocks);
        state.iter().map(|word| format!("{word:08x}")).collect()
    }

    /// The stored form: the hash state at the last 64-byte boundary hashed.
    pub(crate) fn state_hex(&self) -> String {
        self.state
            .iter()
            .map(|word| format!("{word:08x}"))
            .collect()
    }
}

/// The head and tail windows of the bytes consumed so far.
#[derive(Clone)]
pub(crate) struct WindowEvidence {
    /// The first bytes while fewer than [`WINDOW`] have been consumed, then only their digest.
    head: Head,
    /// The last [`WINDOW`] bytes consumed.
    tail: VecDeque<u8>,
}

#[derive(Clone)]
enum Head {
    Filling(Vec<u8>),
    Sealed(String),
}

impl WindowEvidence {
    pub(crate) fn new() -> Self {
        Self {
            head: Head::Filling(Vec::new()),
            tail: VecDeque::new(),
        }
    }

    pub(crate) fn update(&mut self, data: &[u8]) {
        if let Head::Filling(head) = &mut self.head {
            let wanted = (WINDOW as usize - head.len()).min(data.len());
            head.extend_from_slice(&data[..wanted]);
            if head.len() == WINDOW as usize {
                self.head = Head::Sealed(sha256_hex(head));
            }
        }
        self.tail.extend(data);
        let over = self.tail.len().saturating_sub(WINDOW as usize);
        self.tail.drain(..over);
    }

    pub(crate) fn head_sha256(&self) -> String {
        match &self.head {
            Head::Filling(head) => sha256_hex(head),
            Head::Sealed(digest) => digest.clone(),
        }
    }

    pub(crate) fn tail_sha256(&self) -> String {
        let (front, back) = self.tail.as_slices();
        let mut hasher = Sha256::new();
        hasher.update(front);
        hasher.update(back);
        hex(&hasher.finalize())
    }
}

/// What a checkpoint stores about its committed prefix.
pub(crate) struct CommittedPrefix<'a> {
    pub(crate) offset: u64,
    pub(crate) prefix_sha256: &'a str,
    pub(crate) hash_state: &'a str,
    pub(crate) head_window_sha256: &'a str,
    pub(crate) tail_window_sha256: &'a str,
}

/// A verified prefix, ready to continue hashing at the checkpoint.
pub(crate) struct ResumedPrefix {
    pub(crate) hasher: ResumableSha256,
    pub(crate) windows: WindowEvidence,
}

/// Why a committed prefix no longer matches.
pub(crate) enum PrefixMismatch {
    /// The head or tail window differs: the source was replaced or rewritten.
    Window { actual_sha256: String },
    /// The windows match but the stored hash state does not reproduce the stored digest.
    Digest { actual_sha256: String },
}

/// Check a committed prefix by reading only its two windows. The caller has already checked that
/// the source is at least `offset` bytes long.
pub(crate) fn verify_committed_prefix(
    reader: &mut dyn SeekableEventSource,
    committed: &CommittedPrefix<'_>,
) -> Result<std::result::Result<ResumedPrefix, PrefixMismatch>> {
    let head_len = committed.offset.min(WINDOW);
    let head = read_range(reader, 0, head_len)?;
    let head_sha = sha256_hex(&head);
    if head_sha != committed.head_window_sha256 {
        return Ok(Err(PrefixMismatch::Window {
            actual_sha256: head_sha,
        }));
    }
    let tail_start = committed.offset - committed.offset.min(WINDOW);
    let tail = if tail_start == 0 {
        head.clone()
    } else {
        read_range(reader, tail_start, committed.offset - tail_start)?
    };
    let tail_sha = sha256_hex(&tail);
    if tail_sha != committed.tail_window_sha256 {
        return Ok(Err(PrefixMismatch::Window {
            actual_sha256: tail_sha,
        }));
    }
    let boundary = committed.offset / 64 * 64;
    let Some(mut hasher) = ResumableSha256::resume(committed.hash_state, boundary) else {
        return Ok(Err(PrefixMismatch::Digest {
            actual_sha256: String::new(),
        }));
    };
    hasher.update(&tail[(boundary - tail_start) as usize..]);
    let actual = hasher.digest_hex();
    if actual != committed.prefix_sha256 {
        return Ok(Err(PrefixMismatch::Digest {
            actual_sha256: actual,
        }));
    }
    let mut windows = WindowEvidence::new();
    if committed.offset >= WINDOW {
        windows.head = Head::Sealed(head_sha);
        windows.tail = tail.into();
    } else {
        windows.update(&head);
    }
    Ok(Ok(ResumedPrefix { hasher, windows }))
}

/// Hash the first `length` bytes of the source in full (the explicit, on-demand verification).
pub(crate) fn full_prefix_sha256(
    reader: &mut dyn SeekableEventSource,
    length: u64,
) -> Result<String> {
    let mut hasher = ResumableSha256::new();
    hash_range(reader, 0, length, |bytes| hasher.update(bytes))?;
    Ok(hasher.digest_hex())
}

/// The hasher and windows for a prefix hashed from the start (first capture, provisional sources).
pub(crate) fn hash_prefix_from_start(
    reader: &mut dyn SeekableEventSource,
    length: u64,
) -> Result<ResumedPrefix> {
    let mut hasher = ResumableSha256::new();
    let mut windows = WindowEvidence::new();
    hash_range(reader, 0, length, |bytes| {
        hasher.update(bytes);
        windows.update(bytes);
    })?;
    Ok(ResumedPrefix { hasher, windows })
}

fn hash_range(
    reader: &mut dyn SeekableEventSource,
    start: u64,
    length: u64,
    mut visit: impl FnMut(&[u8]),
) -> Result<()> {
    reader
        .seek(SeekFrom::Start(start))
        .map_err(|error| CgError::Engine(error.to_string()))?;
    let mut remaining = length;
    let mut buffer = [0_u8; 16 * 1024];
    while remaining > 0 {
        let wanted = remaining.min(buffer.len() as u64) as usize;
        let read = reader
            .read(&mut buffer[..wanted])
            .map_err(|error| CgError::Engine(error.to_string()))?;
        if read == 0 {
            return Err(CgError::Engine(
                "source ended while verifying its committed prefix".into(),
            ));
        }
        visit(&buffer[..read]);
        remaining -= read as u64;
    }
    Ok(())
}

fn read_range(reader: &mut dyn SeekableEventSource, start: u64, length: u64) -> Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(length as usize);
    hash_range(reader, start, length, |chunk| {
        bytes.extend_from_slice(chunk)
    })?;
    Ok(bytes)
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_resumable_hash_equals_sha256_across_any_split_and_resume() {
        let data = (0..10_000_u32)
            .flat_map(|n| n.to_le_bytes())
            .collect::<Vec<_>>();
        for split in [
            0, 1, 55, 56, 57, 63, 64, 65, 119, 120, 127, 128, 129, 1000, 39_999, 40_000,
        ] {
            let mut first = ResumableSha256::new();
            first.update(&data[..split]);
            assert_eq!(
                first.digest_hex(),
                sha256_hex(&data[..split]),
                "prefix split {split}"
            );
            let boundary = split as u64 / 64 * 64;
            let mut resumed = ResumableSha256::resume(&first.state_hex(), boundary).unwrap();
            resumed.update(&data[boundary as usize..]);
            assert_eq!(resumed.digest_hex(), sha256_hex(&data), "split {split}");
            first.update(&data[split..]);
            assert_eq!(first.digest_hex(), sha256_hex(&data), "split {split}");
        }
    }

    #[test]
    fn the_windows_hold_the_head_and_the_last_window_of_the_stream() {
        let data = (0..200_000_u32).map(|n| n as u8).collect::<Vec<_>>();
        let mut windows = WindowEvidence::new();
        for chunk in data.chunks(7_777) {
            windows.update(chunk);
        }
        assert_eq!(windows.head_sha256(), sha256_hex(&data[..WINDOW as usize]));
        assert_eq!(
            windows.tail_sha256(),
            sha256_hex(&data[data.len() - WINDOW as usize..])
        );
    }
}
