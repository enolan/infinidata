use core::fmt::Debug;
use memmap::Mmap;
use numpy as np;
use numpy::array::*;
use numpy::PyUntypedArray;
use pyo3::prelude::*;
use pyo3::types::*;
use rkyv::ser::serializers::{
    AllocScratch, CompositeSerializer, FallbackScratch, HeapScratch, SharedSerializeMap,
    WriteSerializer,
};
use rkyv::ser::Serializer;
use rkyv::validation::validators::{check_archived_root, DefaultValidator};
use rkyv::{Archive, CheckBytes, Deserialize, Serialize};
use std::fs::File;
use uuid::Uuid;

/// Our top level module
#[pymodule]
fn infinidata(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(table_desc_from_dict, m)?)?;
    m.add_function(wrap_pyfunction!(column_storage_from_pyarray, m)?)?;
    Ok(())
}

/// Possible types of data in a column
#[repr(u8)]
#[derive(Archive, Copy, Clone, Debug, Deserialize, Serialize)]
#[archive(check_bytes)]
#[archive_attr(derive(Debug))]
enum DType {
    F32,
    I32,
    I64,
    FixedLengthString(usize),
}

/// Column definition
#[derive(Archive, Clone, Debug, Deserialize, Serialize)]
#[archive(check_bytes)]
#[archive_attr(derive(Debug))]
struct ColumnDesc {
    name: String,
    dtype: DType,
    dims: Vec<usize>,
}

/// Table definition
#[derive(Archive, Clone, Debug, Deserialize, Serialize)]
#[archive(check_bytes)]
#[archive_attr(derive(Debug))]
struct TableDesc {
    uuid: Uuid,
    columns: Vec<ColumnDesc>,
}

/// Concrete backing storage for a single column
#[derive(Archive, Clone, Debug, Deserialize, Serialize)]
#[archive(check_bytes)]
#[archive_attr(derive(Debug))]
enum ColumnStorage {
    F32(Vec<f32>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    FixedLengthString(Vec<String>),
}

/// Concrete backing storage for a whole table
#[derive(Archive, Clone, Debug, Deserialize, Serialize)]
#[archive(check_bytes)]
#[archive_attr(derive(Debug))]
struct TableStorage {
    uuid: Uuid,
    // Column order matches TableDesc, lengths need to match each other
    columns: Vec<ColumnStorage>,
}

/// A view onto a table
#[derive(Archive, Clone, Debug, Deserialize, Serialize)]
#[archive(check_bytes)]
#[archive_attr(derive(Debug))]
struct TableView {
    uuid: Uuid,
    desc_uuid: Uuid,
    index_mapping: IndexMapping,
}

/// A mapping from the indices in a view to the indices in a table
#[derive(Archive, Clone, Debug, Deserialize, Serialize)]
#[archive(check_bytes)]
#[archive_attr(derive(Debug))]
enum IndexMapping {
    /// Map each index to that index in the referenced TableStorage
    Storage(Uuid),
    /// Map indices to the indices in the referenced TableViews, sequentially
    Concat(Vec<Uuid>),
    /// Map indices to a single TableView, one by one
    Indices(Uuid, Vec<usize>),
    /// Map to a range of indices in a TableView
    Range {
        table_uuid: Uuid,
        start: usize,
        end: usize,
        step: usize,
    },
}

#[pyfunction]
fn table_desc_from_dict(py: Python<'_>, dict: &PyDict) -> PyResult<()> {
    let uuid = Uuid::new_v4();
    let mut columns = Vec::new();
    let mut data_len = None;
    for (key, value) in dict.iter() {
        let key = key.extract::<String>()?;
        let value = value.downcast::<PyUntypedArray>()?;
        let dtype = dtype_from_pyarray(py, value)?;
        if value.shape().len() < 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Unsupported shape, must be at least 1D",
            ));
        }
        match data_len {
            None => data_len = Some(value.shape()[0]),
            Some(len) => {
                if len != value.shape()[0] {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "All columns must have the same length",
                    ));
                }
            }
        }
        let column = ColumnDesc {
            name: key.clone(),
            dtype,
            dims: value.shape()[1..].to_vec(),
        };
        columns.push(column);
    }
    let td = TableDesc { uuid, columns };
    write_serialize(&td, "table_desc.bin")?;
    let td_archived = load_archived::<TableDesc>("table_desc.bin").map_err(|err| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Error loading table desc: {}", err))
    })?;
    println!("loaded TableDesc: {:?}", *td_archived);
    Ok(())
}

/// Get the dtype from a numpy array
fn dtype_from_pyarray(py: Python<'_>, array: &PyUntypedArray) -> PyResult<DType> {
    let dtype_py = array.dtype();
    if dtype_py.is_equiv_to(np::dtype::<f32>(py)) {
        Ok(DType::F32)
    } else if dtype_py.is_equiv_to(np::dtype::<i32>(py)) {
        Ok(DType::I32)
    } else if dtype_py.is_equiv_to(np::dtype::<i64>(py)) {
        Ok(DType::I64)
    } else {
        let kind = dtype_py.kind();
        if kind == b'U' {
            let length = dtype_py.itemsize() as usize / 4;
            Ok(DType::FixedLengthString(length))
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unsupported dtype {:?}",
                dtype_py
            )))
        }
    }
}

/// Force a numpy array to be contiguous, potentially copying it
fn make_contiguous<'py>(py: Python<'py>, array: &'py PyUntypedArray) -> &'py PyUntypedArray {
    if array.is_c_contiguous() {
        array
    } else {
        let np = py.import("numpy").unwrap();
        let fun = np.getattr("ascontiguousarray").unwrap();
        let contiguous_array = fun.call1((array,)).unwrap();
        let contiguous_array = contiguous_array.downcast::<PyUntypedArray>().unwrap();
        contiguous_array
    }
}

#[pyfunction]
fn column_storage_from_pyarray(py: Python<'_>, array: &PyUntypedArray) -> PyResult<()> {
    let dtype = dtype_from_pyarray(py, array)?;
    let array = make_contiguous(py, array);
    let storage = match dtype {
        DType::F32 => {
            let arr = array.downcast::<PyArrayDyn<f32>>().unwrap();
            let data = arr.to_vec().unwrap();
            ColumnStorage::F32(data)
        }
        DType::I32 => {
            let arr = array.downcast::<PyArrayDyn<i32>>().unwrap();
            let data = arr.to_vec().unwrap();
            ColumnStorage::I32(data)
        }
        DType::I64 => {
            let arr = array.downcast::<PyArrayDyn<i64>>().unwrap();
            let data = arr.to_vec().unwrap();
            ColumnStorage::I64(data)
        }
        DType::FixedLengthString(_str_max_len) => {
            let total_len: usize = array.shape().iter().product();
            let arr = array.call_method1("reshape", (total_len,)).unwrap();
            let mut strs = Vec::with_capacity(total_len);
            for str in arr.iter().unwrap() {
                let str = str.unwrap().extract::<String>().unwrap();
                println!("str: {:?}", str);
                strs.push(str);
            }
            ColumnStorage::FixedLengthString(strs)
        }
    };
    println!("storage from array: {:?}", storage);
    Ok(())
}

type FileSerializer = CompositeSerializer<
    WriteSerializer<File>,
    FallbackScratch<HeapScratch<4096>, AllocScratch>,
    SharedSerializeMap,
>;

/// Write a Serialize type to a file
fn write_serialize<T>(data: &T, path: &str) -> Result<(), std::io::Error>
where
    T: Serialize<FileSerializer>,
{
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(path)?;
    let write_ser = WriteSerializer::new(file);
    let scratch = FallbackScratch::new(HeapScratch::new(), AllocScratch::new());
    let mut ser = CompositeSerializer::new(write_ser, scratch, SharedSerializeMap::new());
    ser.serialize_value(data).map_or_else(
        |e| Err(std::io::Error::new(std::io::ErrorKind::Other, e)),
        |_| Ok(()),
    )
}

mod mmap_archived {
    use super::*;

    /// An mmapped archived type that has been checked for validity.
    pub struct MmapArchived<T> {
        mmap: Mmap,
        _phantom: std::marker::PhantomData<T>,
    }

    impl<T> MmapArchived<T>
    where
        T: Archive,
        for<'a> T::Archived: CheckBytes<DefaultValidator<'a>>,
    {
        pub fn new(mmap: Mmap) -> Result<Self, String> {
            // There are situations where skipping the check is valid, if profiling shows it
            // matters, we can add an unsafe function to skip the check.
            let buf = &mmap[..];
            let check_res = check_archived_root::<T>(buf);
            match check_res {
                Ok(_) => {
                    // The result has a reference to the buffer, so we need to drop it before we
                    // can move the mmap into the struct.
                    drop(check_res);
                    Ok(MmapArchived {
                        mmap,
                        _phantom: std::marker::PhantomData,
                    })
                }
                Err(e) => Err(format!("CheckBytes error: {}", e)),
            }
        }
    }

    impl<T> std::ops::Deref for MmapArchived<T>
    where
        T: Archive,
    {
        type Target = T::Archived;

        #[inline]
        fn deref(&self) -> &Self::Target {
            let buf = &self.mmap[..];
            unsafe { rkyv::archived_root::<T>(buf) }
        }
    }
}
use mmap_archived::*;

/// Load an archived type from a file with mmap
fn load_archived<T>(path: &str) -> Result<MmapArchived<T>, String>
where
    T: Archive,
    for<'a> T::Archived: CheckBytes<DefaultValidator<'a>>,
{
    let file = File::open(path).map_err(|e| format!("Error opening file: {}", e))?;
    let mmap = unsafe { Mmap::map(&file).map_err(|e| format!("Error mmaping file: {}", e))? };
    MmapArchived::new(mmap)
}
