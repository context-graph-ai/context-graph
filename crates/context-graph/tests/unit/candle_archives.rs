//! Candle archive readers preserve names, values, and typed failures.

use candle_core::npy::NpzTensors;
use candle_core::pickle::{PthTensors, read_pth_tensor_info};
use candle_core::{DType, Device, Error, Result, Tensor};
use std::borrow::Cow;
use std::path::Path;
use zip::result::ZipError;

const PTH: &[u8] = include_bytes!("../../../../vendor/candle-core/tests/test.pt");

fn write_npz(path: &Path, name: &str) -> Result<()> {
    let tensor = Tensor::new(&[2u32, 4, 6], &Device::Cpu)?;
    Tensor::write_npz(&[(name, &tensor)], path)
}

fn assert_npz(path: &Path, name: &str) -> Result<()> {
    let loaded = Tensor::read_npz(path)?;
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0].0, name);
    assert_eq!(loaded[0].1.dtype(), DType::U32);
    assert_eq!(loaded[0].1.to_vec1::<u32>()?, vec![2, 4, 6]);
    let lazy = NpzTensors::new(path)?;
    assert_eq!(
        lazy.names().iter().map(|n| n.as_str()).collect::<Vec<_>>(),
        vec![name]
    );
    let tensor = lazy.get(name)?.expect("the tensor must be indexed by name");
    assert_eq!(tensor.dtype(), DType::U32);
    assert_eq!(tensor.to_vec1::<u32>()?, vec![2, 4, 6]);
    assert!(lazy.get("absent")?.is_none());
    Ok(())
}

fn central_header(bytes: &[u8]) -> usize {
    bytes
        .windows(4)
        .rposition(|w| w == b"PK\x01\x02")
        .expect("the tiny archive must have a central directory")
}

fn assert_truncated_filename<T>(result: Result<T>) {
    match result {
        Err(Error::Zip(ZipError::InvalidArchive(message))) => {
            assert_eq!(
                message,
                "Variable-length field extends beyond file boundary"
            );
        }
        Err(other) => panic!("expected a typed ZIP filename read failure, got {other:?}"),
        Ok(_) => panic!("a truncated filename must be refused"),
    }
}

/// Eager and lazy NPZ reads preserve borrowed UTF-8 filenames and exact tensor values.
#[test]
fn npz_utf8_names_and_values_roundtrip() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("utf8.npz");
    write_npz(&path, "données")?;
    let mut archive = zip::ZipArchive::new(std::fs::File::open(&path)?)?;
    let entry = archive.by_index(0)?;
    assert!(matches!(entry.name()?, Cow::Borrowed("données.npy")));
    assert_npz(&path, "données")
}

/// Eager and lazy NPZ reads preserve owned CP437 filenames and exact tensor values.
#[test]
fn npz_cp437_names_and_values_roundtrip() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("cp437.npz");
    write_npz(&path, "a")?;
    let mut bytes = std::fs::read(&path)?;
    let central = central_header(&bytes);
    let local = u32::from_le_bytes(bytes[central + 42..central + 46].try_into().unwrap()) as usize;
    assert_eq!(&bytes[central + 46..central + 51], b"a.npy");
    assert_eq!(&bytes[local + 30..local + 35], b"a.npy");
    assert_eq!(
        u16::from_le_bytes(bytes[central + 8..central + 10].try_into().unwrap()) & 0x800,
        0
    );
    bytes[central + 46] = 0x82;
    bytes[local + 30] = 0x82;
    std::fs::write(&path, bytes)?;
    let mut archive = zip::ZipArchive::new(std::fs::File::open(&path)?)?;
    let entry = archive.by_index(0)?;
    assert!(matches!(entry.name()?, Cow::Owned(name) if name == "é.npy"));
    assert_npz(&path, "é")
}

/// Both NPZ readers return a typed ZIP error when filename metadata runs past EOF.
#[test]
fn npz_truncated_filename_returns_typed_zip_error() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("truncated.npz");
    write_npz(&path, "a")?;
    let mut bytes = std::fs::read(&path)?;
    let central = central_header(&bytes);
    bytes[central + 28..central + 30].copy_from_slice(&u16::MAX.to_le_bytes());
    std::fs::write(&path, bytes)?;
    assert_truncated_filename(Tensor::read_npz(&path));
    assert_truncated_filename(NpzTensors::new(&path));
    Ok(())
}

/// PTH filename iteration reaches real tensor metadata and loads the published tiny fixture.
#[test]
fn pth_filename_iteration_preserves_tensor_metadata_and_values() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("tensor.pt");
    std::fs::write(&path, PTH)?;
    let infos = read_pth_tensor_info(&path, false, None)?;
    assert_eq!(infos.len(), 1);
    let info = &infos[0];
    assert_eq!(info.name, "test");
    assert_eq!(info.dtype, DType::I64);
    assert_eq!(info.layout.dims(), &[2, 4]);
    assert_eq!(info.path, "test/data/0");
    assert_eq!(info.storage_size, 8);
    let lazy = PthTensors::new(&path, None)?;
    assert_eq!(lazy.tensor_infos().len(), 1);
    let tensor = lazy.get("test")?.expect("the PTH tensor must be indexed");
    assert_eq!(
        tensor.to_vec2::<i64>()?,
        vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]]
    );
    Ok(())
}

/// PTH readers preserve the typed ZIP failure for a truncated central-directory filename.
#[test]
fn pth_truncated_filename_returns_typed_zip_error() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("truncated.pt");
    let mut bytes = PTH.to_vec();
    let central = central_header(&bytes);
    bytes[central + 28..central + 30].copy_from_slice(&u16::MAX.to_le_bytes());
    std::fs::write(&path, bytes)?;
    assert_truncated_filename(read_pth_tensor_info(&path, false, None));
    assert_truncated_filename(PthTensors::new(&path, None));
    Ok(())
}

/// NPZ and PTH entry points reject a non-archive with their existing typed ZIP error.
#[test]
fn non_archives_return_typed_zip_errors() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("invalid.zip");
    std::fs::write(&path, b"not a zip archive")?;
    assert!(matches!(Tensor::read_npz(&path), Err(Error::Zip(_))));
    assert!(matches!(NpzTensors::new(&path), Err(Error::Zip(_))));
    assert!(matches!(
        read_pth_tensor_info(&path, false, None),
        Err(Error::Zip(_))
    ));
    assert!(matches!(PthTensors::new(&path, None), Err(Error::Zip(_))));
    Ok(())
}
