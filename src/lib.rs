use core::fmt::Debug;
use memmap::Mmap;
use numpy as np;
use numpy::array::*;
use numpy::PyUntypedArray;
use pyo3::prelude::*;
use pyo3::types::*;
use rkyv::{Archive, CheckBytes, Deserialize, Serialize};
use rkyv::ser::Serializer;
use rkyv::ser::serializers::{
    AllocScratch, CompositeSerializer, FallbackScratch, HeapScratch, SharedSerializeMap,
    WriteSerializer,
};
use rkyv::validation::validators::{check_archived_root, DefaultValidator};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use uuid::Uuid;

/// Our top level module
#[pymodule]
fn infinidata(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<TableViewMem>()?;
    Ok(())
}

/// Possible types of data in a column
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

/// A TableView along with the associated in-RAM metadata
#[pyclass(name = "TableView")]
#[derive(Debug)]
struct TableViewMem {
    view: MmapArchived<TableView>,
    desc: Arc<MmapArchived<TableDesc>>,
    // If the IndexMapping is Storage. Storages can't be referenced by multiple views, so this
    // doesn't need to be reference counted.
    storage: Option<MmapArchived<TableStorage>>,
    // OTOH TableViews *can* be referenced by multiple views, so they get Arcs.
    concat_views: Option<Vec<Arc<MmapArchived<TableView>>>>, // If it's Concat
    referenced_views: Option<Arc<MmapArchived<TableView>>>,  // If it's Indices or Range
}

#[pymethods]
impl TableViewMem {
    #[new]
    fn new(dict: &PyDict) -> Self {
        Python::with_gil(|py| {
            let (desc, storage) = table_desc_and_columns_from_dict(py, dict).unwrap();
            let view = TableView {
                uuid: Uuid::new_v4(),
                desc_uuid: desc.uuid,
                index_mapping: IndexMapping::Storage(storage.uuid),
            };
            let path: PathBuf = format!("TableView-{}.bin", view.uuid).into();
            let file = write_serialize(&view, &path).unwrap();
            let view_archived = load_archived::<TableView>(file, &path).unwrap();
            let desc = Arc::new(desc);
            let tv = TableViewMem {
                view: view_archived,
                desc,
                storage: Some(storage),
                concat_views: None,
                referenced_views: None,
            };
            println!("TableViewMem: {:?}", tv);
            tv
        })
    }
}

fn table_desc_and_columns_from_dict(
    py: Python<'_>,
    dict: &PyDict,
) -> PyResult<(MmapArchived<TableDesc>, MmapArchived<TableStorage>)> {
    let column_cnt = dict.len();
    let mut column_descs = Vec::with_capacity(column_cnt);
    let mut column_storages = Vec::with_capacity(column_cnt);
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
        let desc = ColumnDesc {
            name: key.clone(),
            dtype,
            dims: value.shape()[1..].to_vec(),
        };
        column_descs.push(desc);
        let storage = column_storage_from_pyarray(py, value)?;
        column_storages.push(storage);
    }
    let td_uuid = Uuid::new_v4();
    let td = TableDesc {
        uuid: td_uuid,
        columns: column_descs,
    };
    let path: PathBuf = format!("TableDesc-{}.bin", td_uuid).into();
    let file = write_serialize(&td, &path)?;
    let td_archived = load_archived::<TableDesc>(file, &path).map_err(|err| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Error loading table desc: {}", err))
    })?;

    let ts_uuid = Uuid::new_v4();
    let ts = TableStorage {
        uuid: td_uuid,
        columns: column_storages,
    };
    let path: PathBuf = format!("TableStorage-{}.bin", ts_uuid).into();
    let file = write_serialize(&ts, &path)?;
    let ts_archived = load_archived::<TableStorage>(file, &path).map_err(|err| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Error loading table storage: {}", err))
    })?;

    Ok((td_archived, ts_archived))
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

fn column_storage_from_pyarray(py: Python<'_>, array: &PyUntypedArray) -> PyResult<ColumnStorage> {
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
            // For some reason there's no Rust interface to reshape on PyUntypedArrays
            let arr = array.call_method1("reshape", (total_len,)).unwrap();
            let mut strs = Vec::with_capacity(total_len);
            for str in arr.iter().unwrap() {
                let str = str.unwrap().extract::<String>().unwrap();
                strs.push(str);
            }
            assert_eq!(strs.len(), total_len);
            ColumnStorage::FixedLengthString(strs)
        }
    };
    println!("storage from array: {:?}", &storage);
    Ok(storage)
}

type FileSerializer = CompositeSerializer<
    WriteSerializer<File>,
    FallbackScratch<HeapScratch<4096>, AllocScratch>,
    SharedSerializeMap,
>;

/// Write a Serialize type to a file
fn write_serialize<T>(data: &T, path: &Path) -> Result<File, std::io::Error>
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
    )?;
    // file was moved into ser, so we need to move it back out
    let file = ser.into_components().0.into_inner();
    let mut perms = file.metadata()?.permissions();
    perms.set_readonly(true);
    file.set_permissions(perms)?;
    Ok(file)
}

mod mmap_archived {
    use super::*;

    /// An mmapped archived type that has been checked for validity.
    #[derive(Debug)]
    pub struct MmapArchived<T> {
        mmap: Mmap,
        fname: PathBuf,
        delete_on_drop: bool,
        _phantom: std::marker::PhantomData<T>,
    }

    impl<T> MmapArchived<T>
    where
        T: Archive,
        for<'a> T::Archived: CheckBytes<DefaultValidator<'a>>,
    {
        pub fn new(file: File, fname: &Path, delete_on_drop: bool) -> Result<Self, String> {
            let fname = fname
                .canonicalize()
                .map_err(|e| format!("Error canonicalizing path: {}", e))?;
            let mmap =
                unsafe { Mmap::map(&file).map_err(|e| format!("Error mmaping file: {}", e))? };
            // There are situations where skipping the check is valid, if profiling shows it
            // matters, we can add an unsafe function to skip the check.
            let check_res = check_archived_root::<T>(&mmap[..]);
            match check_res {
                Ok(_) => {
                    // The result has a reference to the buffer, so we need to drop it before we
                    // can move the mmap into the struct.
                    drop(check_res);
                    Ok(MmapArchived {
                        mmap,
                        fname,
                        delete_on_drop,
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

        #[inline(always)]
        fn deref(&self) -> &Self::Target {
            unsafe { rkyv::archived_root::<T>(&self.mmap[..]) }
        }
    }

    impl<T> Drop for MmapArchived<T> {
        fn drop(&mut self) {
            if self.delete_on_drop {
                match std::fs::remove_file(&self.fname) {
                    Ok(_) => (),
                    Err(e) => println!("Warning: error deleting file {:?}: {}", &self.fname, e),
                }
            }
        }
    }
}
use mmap_archived::*;

/// Load an archived type from a file with mmap
fn load_archived<T>(file: File, path: &Path) -> Result<MmapArchived<T>, String>
where
    T: Archive,
    for<'a> T::Archived: CheckBytes<DefaultValidator<'a>>,
{
    MmapArchived::new(file, path, true)
}
