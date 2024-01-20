use core::fmt::Debug;
use crossbeam_channel::Receiver;
use memmap::Mmap;
use mvar::Mvar;
use numpy as np;
use numpy::array::*;
use numpy::PyUntypedArray;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::*;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rand::SeedableRng;
use rkyv::ser::serializers::{
    AllocScratch, CompositeSerializer, FallbackScratch, HeapScratch, SharedSerializeMap,
    WriteSerializer,
};
use rkyv::ser::Serializer;
use rkyv::validation::validators::{check_archived_root, DefaultValidator};
use rkyv::{Archive, CheckBytes, Deserialize, Serialize};
use std::any::type_name;
use std::cmp::{min, Ordering};
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;
use uuid::Uuid;

/// Our top level module
#[pymodule]
fn infinidata(_py: Python, m: &PyModule) -> PyResult<()> {
    unsafe { setup_tmpdir_path() }; // It's safe to invoke this because we have the GIL here.
    m.add_class::<TableViewMem>()?;
    Ok(())
}

/// Possible types of data in a column
#[derive(Archive, Copy, Clone, Debug, Deserialize, Serialize)]
#[archive(check_bytes)]
#[archive_attr(derive(Debug, Eq, PartialEq))]
enum DType {
    F32,
    I32,
    I64,
    UString,
}

/// Column definition
#[derive(Archive, Clone, Debug, Deserialize, Serialize)]
#[archive(check_bytes)]
#[archive_attr(derive(Debug, Eq, PartialEq))]
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
    UString(Vec<String>),
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
        // These are signed sizes to support negative steps and when the stop index is -1, which
        // means we cover index 0. The start index can never be negative, that would be OOB.
        start: usize,
        stop: isize,
        step: isize,
    },
}

/// A view into a table. A table is a collection of columns, each of which has a name along with a
/// dtype and a shape for the entries. E.g.:
///
/// tbl_dict = {
///   "foo": np.arange(45*16*2, dtype=np.float32).reshape((45,16,2)),
///   "bar": np.arange(45, dtype=np.int64),
///   "baz": np.array(["hello"] * 45)
/// }
/// tbl = infinidata.TableView(tbl_dict)
///
/// Here, tbl_dict is a dictionary mapping column names to NumPy arrays. We use it to construct
/// a TableView. The columns are "foo", "bar", and "baz", with dtypes float32, int64, and string,
/// and shapes (16, 2), (), and (). The table has 45 rows. The rows of a TableView can be accessed
/// by subscripting with []: tv[5] gets row 5, tv[2:5] gets rows 2, 3, and 4, and
/// tv[np.array([1, 3, 5])] gets rows 1, 3, and 5. You can also use a slice with a step size:
/// tv[1:10:2] gets rows 1, 3, 5, 7, and 9.
#[pyclass(name = "TableView")]
#[derive(Clone, Debug)]
struct TableViewMem {
    // We separate the TableView Rust struct from the TableViewMem Rust struct, and only expose the
    // latter to Python. TableView is Archive, and is what goes on disk, TableViewMem has access
    // to the underlying TableView and TableDesc via mmap, and carries various info used to resolve
    // access to the underlying data as well as create new TableViews.
    view: Arc<MmapArchived<TableView>>,
    desc: Arc<MmapArchived<TableDesc>>,
    storage: Option<Arc<MmapArchived<TableStorage>>>, // If the IndexMapping is Storage
    concat_views: Option<(Vec<TableViewMem>, Vec<usize>)>, // If it's Concat
    referenced_view: Option<Box<TableViewMem>>,       // If it's Indices or Range
    /// Columns that we're looking at - it's possible to ignore columns. For now this is a purely
    /// runtime effect and ignoring a column doesn't cause the column to be removed from the
    /// storage. Mostly you want to ignore columns to save disk I/O, not disk space.
    live_columns: Vec<usize>,
}

#[pymethods]
impl TableViewMem {
    #[new]
    fn new(dict: &PyDict) -> Self {
        let py = dict.py();
        let (desc, storage) = table_desc_and_columns_from_dict(py, dict).unwrap(); // FIXME result
        let view = TableView {
            uuid: Uuid::new_v4(),
            desc_uuid: desc.uuid,
            index_mapping: IndexMapping::Storage(storage.uuid),
        };
        let view_archived = unsafe { make_mmapped(&view) };
        let ncols = desc.columns.len();
        TableViewMem {
            view: Arc::new(view_archived),
            desc: Arc::new(desc),
            storage: Some(Arc::new(storage)),
            concat_views: None,
            referenced_view: None,
            live_columns: (0..ncols).collect(),
        }
    }

    #[pyo3(name = "__getitem__")]
    fn get_item(&self, index: &PyAny) -> PyResult<Py<PyDict>> {
        let column_descs: Vec<(usize, &ArchivedColumnDesc)> = self
            .live_columns
            .iter()
            .map(|&col| (col, &self.desc.columns[col]))
            .collect();
        let py = index.py();
        let out = PyDict::new(py);
        if let Ok(index) = index.extract::<usize>() {
            if index >= self.len() {
                return Err(pyo3::exceptions::PyIndexError::new_err(
                    "Index out of bounds",
                ));
            }
            for (col, col_desc) in column_descs {
                let col_name = col_desc.name.to_string();
                let arr: &PyUntypedArray = match col_desc.dtype {
                    ArchivedDType::F32 => {
                        let iter = self.get_f32_column_at_idx(col, index);
                        np::PyArray::from_iter(py, iter).downcast().unwrap()
                    }
                    ArchivedDType::I32 => {
                        let iter = self.get_i32_column_at_idx(col, index);
                        np::PyArray::from_iter(py, iter).downcast().unwrap()
                    }
                    ArchivedDType::I64 => {
                        let iter = self.get_i64_column_at_idx(col, index);
                        np::PyArray::from_iter(py, iter).downcast().unwrap()
                    }
                    ArchivedDType::UString => {
                        let iter = self.get_string_column_at_idx(col, index);
                        let strings = iter.collect::<Vec<String>>();
                        let string_list = PyList::new(py, strings);
                        let np = py.import(intern!(py, "numpy")).unwrap();
                        let fun = np.getattr(intern!(py, "array")).unwrap();
                        fun.call1((string_list,)).unwrap().downcast().unwrap()
                    }
                };
                let dims: &[usize] = &col_desc
                    .dims
                    .iter()
                    .map(|&d| d as usize)
                    .collect::<Vec<usize>>();
                if !dims.is_empty() {
                    out.set_item(col_name, reshape_pyuntypedarray(py, arr, dims).unwrap())
                        .unwrap();
                } else {
                    out.set_item(col_name, arr.get_item(0).unwrap()).unwrap();
                }
            }
            Ok(out.into())
        } else if let Ok(slice) = index.downcast::<PySlice>() {
            let slice_idxs = slice.indices(self.len() as i64)?;
            // This has the Python semantics where getting a slice is never out of bounds, even if
            // your bounds go past the end of the array. You do get an exception if you try to
            // specify a step size of 0 though.
            for (col, col_desc) in column_descs {
                let col_name = col_desc.name.to_string();
                let arr: &PyUntypedArray = match col_desc.dtype {
                    ArchivedDType::F32 => {
                        let iter = self.get_f32_column_range(col, &slice_idxs);
                        np::PyArray::from_iter(py, iter).downcast().unwrap()
                    }
                    ArchivedDType::I32 => {
                        let iter = self.get_i32_column_range(col, &slice_idxs);
                        np::PyArray::from_iter(py, iter).downcast().unwrap()
                    }
                    ArchivedDType::I64 => {
                        let iter = self.get_i64_column_range(col, &slice_idxs);
                        np::PyArray::from_iter(py, iter).downcast().unwrap()
                    }
                    ArchivedDType::UString => {
                        let iter = self.get_string_column_range(col, &slice_idxs);
                        let strings = iter.collect::<Vec<String>>();
                        let string_list = PyList::new(py, strings);
                        let np = py.import(intern!(py, "numpy")).unwrap();
                        let fun = np.getattr(intern!(py, "array")).unwrap();
                        fun.call1((string_list,)).unwrap().downcast().unwrap()
                    }
                };
                let out_dims = std::iter::once(slice_idxs.slicelength as usize)
                    .chain(col_desc.dims.iter().map(|&d| d as usize))
                    .collect::<Vec<usize>>();
                let arr = reshape_pyuntypedarray(py, arr, &out_dims).unwrap();
                out.set_item(col_name, arr).unwrap();
            }
            Ok(out.into())
        } else if let Ok(idx_array) = index.downcast::<PyArray1<i64>>() {
            todo!("get_item array {idx_array}")
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "Index must be an integer, slice, or NumPy array of integers",
            ))
        }
    }

    #[pyo3(name = "__len__")]
    fn len(&self) -> usize {
        match &self.view.index_mapping {
            ArchivedIndexMapping::Storage(_storage_uuid) => {
                let storage = self
                    .storage
                    .as_ref()
                    .expect("storage not set when IndexMapping is Storage");
                let col_storage = &storage.columns[0];
                let dims_product: usize =
                    self.desc.columns[0].dims.iter().product::<u64>() as usize;
                let elem_len = match col_storage {
                    ArchivedColumnStorage::F32(data) => data.len(),
                    ArchivedColumnStorage::I32(data) => data.len(),
                    ArchivedColumnStorage::I64(data) => data.len(),
                    ArchivedColumnStorage::UString(data) => data.len(),
                };
                elem_len / dims_product
            }
            ArchivedIndexMapping::Concat(_views) => {
                let (_concat_views, cum_lengths) = self
                    .concat_views
                    .as_ref()
                    .expect("concat_views not set when IndexMapping is Concat");
                *cum_lengths.last().unwrap()
            }
            ArchivedIndexMapping::Indices(_view_uuid, indices) => indices.len(),
            ArchivedIndexMapping::Range {
                table_uuid: _,
                start,
                stop,
                step,
            } => ((stop - *start as i64) / step) as usize,
        }
    }

    /// Get the UUID of the TableView
    fn uuid(&self) -> String {
        self.view.uuid.to_string()
    }

    /// Concatenate multiple TableViews together
    #[staticmethod]
    fn concat(views: &PyList) -> PyResult<Self> {
        if views.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Must pass at least one view to concat",
            ));
        }
        let mut concat_views: Vec<TableViewMem> = Vec::with_capacity(views.len());
        let mut view_lens: Vec<usize> = Vec::with_capacity(views.len());
        let all_desc: &ArchivedTableDesc = &views[0].extract::<PyRef<Self>>()?.desc;
        for (i, view) in views.iter().enumerate() {
            let view = view.extract::<PyRef<Self>>()?;
            view_lens.push(view.len());
            if i != 0 && view.desc.columns != all_desc.columns {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "All views must have the same column definitions",
                ));
            }
            concat_views.push(view.clone());
        }
        let table_view = TableView {
            uuid: Uuid::new_v4(),
            desc_uuid: all_desc.uuid,
            index_mapping: IndexMapping::Concat(concat_views.iter().map(|v| v.view.uuid).collect()),
        };

        // Compute cumulative lengths of the constituent views
        let mut cum_lengths = Vec::with_capacity(concat_views.len());
        let mut cum_length = 0;
        for len in &view_lens {
            cum_length += len;
            cum_lengths.push(cum_length);
        }

        let table_view_archived = unsafe { make_mmapped(&table_view) };
        Ok(TableViewMem {
            view: Arc::new(table_view_archived),
            desc: Arc::clone(&views[0].extract::<PyRef<Self>>()?.desc),
            storage: None,
            concat_views: Some((concat_views, cum_lengths)),
            referenced_view: None,
            live_columns: (0..all_desc.columns.len()).collect(),
        })
    }

    /// Make a new TableView from an existing one, remapping the indices either using an index
    /// array or a range. E.g.:
    ///
    /// tv = infinidata.TableView({"foo": np.arange(45, dtype=np.int64)})
    /// tv2 = tv.new_view(np.array([1, 3, 5]))
    /// tv3 = tv.new_view(slice(1, 10, 2))
    /// tv4 = tv.new_view(slice(None, None, -1))
    ///
    /// tv2 is a new view with the 1st, 3rd, and 5th rows of tv. The slice function is equivalent
    /// the [start:stop:step] notation used when subscripting. tv3 is a new view with the 1st, 3rd,
    /// 5th, 7th, and 9th rows of tv. tv4 is a new view with the rows of tv in reverse order.
    fn new_view(&self, mapping: &PyAny) -> PyResult<Self> {
        let py = mapping.py();
        if let Ok(idx_array) = mapping.downcast::<PyArray1<i64>>() {
            let idx_array = make_contiguous(py, idx_array)
                .downcast::<PyArray1<i64>>()
                .unwrap()
                .readonly();
            let idx_vec = idx_array
                .as_slice()
                .unwrap()
                .iter()
                .map(|&i| i as usize)
                .collect::<Vec<usize>>();
            let table_view = TableView {
                uuid: Uuid::new_v4(),
                desc_uuid: self.desc.uuid,
                index_mapping: IndexMapping::Indices(self.view.uuid, idx_vec),
            };
            let table_view_archived = unsafe { make_mmapped(&table_view) };
            Ok(TableViewMem {
                view: Arc::new(table_view_archived),
                desc: Arc::clone(&self.desc),
                storage: None,
                concat_views: None,
                referenced_view: Some(Box::new(self.clone())),
                live_columns: self.live_columns.clone(),
            })
        } else if let Ok(slice) = mapping.downcast::<PySlice>() {
            let slice_idxs = slice.indices(self.len() as i64)?;
            let table_view = TableView {
                uuid: Uuid::new_v4(),
                desc_uuid: self.desc.uuid,
                index_mapping: IndexMapping::Range {
                    table_uuid: self.view.uuid,
                    start: slice_idxs.start as usize,
                    stop: slice_idxs.stop,
                    step: slice_idxs.step,
                },
            };
            let table_view_archived = unsafe { make_mmapped(&table_view) };
            Ok(TableViewMem {
                view: Arc::new(table_view_archived),
                desc: Arc::clone(&self.desc),
                storage: None,
                concat_views: None,
                referenced_view: Some(Box::new(self.clone())),
                live_columns: self.live_columns.clone(),
            })
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "Index must be a slice or NumPy array of integers",
            ))
        }
    }

    /// Shuffles the rows of a TableView. N.b. this will use enough memory to make a complete
    /// index array. An offline approach is possible if you have an absurd number of rows, but not
    /// implemented. The memory is freed after the shuffle is complete - the generated index array
    /// is stored on disk.
    fn shuffle(&self, seed: Option<u64>) -> PyResult<Self> {
        let indices: &mut [usize] = &mut (0..self.len()).collect::<Vec<usize>>();
        match seed {
            None => {
                let mut rng = thread_rng();
                indices.shuffle(&mut rng);
            }
            Some(seed) => {
                let mut rng = StdRng::seed_from_u64(seed);
                indices.shuffle(&mut rng);
            }
        }
        let table_view = TableView {
            uuid: Uuid::new_v4(),
            desc_uuid: self.desc.uuid,
            index_mapping: IndexMapping::Indices(self.view.uuid, indices.to_vec()),
        };
        let table_view_archived = unsafe { make_mmapped(&table_view) };
        Ok(TableViewMem {
            view: Arc::new(table_view_archived),
            desc: Arc::clone(&self.desc),
            storage: None,
            concat_views: None,
            referenced_view: Some(Box::new(self.clone())),
            live_columns: self.live_columns.clone(),
        })
    }

    /// Iterate over the rows of a TableView in batches.
    fn batch_iter(
        &self,
        batch_size: usize,
        drop_last_batch: bool,
        threads: Option<u32>,
        readahead: Option<u32>,
    ) -> PyResult<BatchIter> {
        let threads = threads.unwrap_or(1);
        let readahead = readahead.unwrap_or(1);
        if drop_last_batch && batch_size > self.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "batch_size must be <= the number of rows in the table if drop_last_batch is true",
            ));
        }
        if threads < 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "threads must be >= 1",
            ));
        }
        if readahead < 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "readahead must be >= 1",
            ));
        }
        if threads > readahead {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "threads must be <= readahead",
            ));
        }
        Ok(BatchIter::new(
            self.clone(),
            batch_size,
            drop_last_batch,
            threads,
            readahead,
        ))
    }

    /// Select the columns to be viewed, returning a new TableView. Subscripting the TableView and
    /// using batch_iter will return dicts with only those columns.
    fn select_columns(&self, columns: &PySet) -> PyResult<Self> {
        let mut live_columns = Vec::new();
        for col in columns.iter() {
            let col = col.extract::<String>()?;
            let col_idx = self
                .desc
                .columns
                .iter()
                .position(|c| c.name == col)
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(format!("No such column: {}", col))
                })?;
            live_columns.push(col_idx);
        }
        Ok(TableViewMem {
            view: Arc::clone(&self.view),
            desc: Arc::clone(&self.desc),
            storage: self.storage.clone(),
            concat_views: self.concat_views.clone(),
            referenced_view: self.referenced_view.clone(),
            live_columns,
        })
    }

    /// Remove rows from the table where a given string column matches an element of a given set.
    /// This materializes the full set of retained indices in memory, so it's not suitable for
    /// obscenely large numbers of rows. An offline approach is possible, but not implemented.
    fn remove_matching_strings(
        &self,
        column: &str,
        strings_to_remove: HashSet<String>,
    ) -> PyResult<Self> {
        let col_idx = self
            .desc
            .columns
            .iter()
            .position(|c| c.name == column)
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!("No such column: {}", column))
            })?;
        match self.desc.columns[col_idx].dtype {
            ArchivedDType::UString => {}
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Column must be a string column",
                ))
            }
        }

        // 50% keep rate seems vaguely reasonable? My actual duplicate blacklists probably keep
        // more like 99% but that seems like an excessive amount of allocation. shrug emoji.
        let mut retained_indices = Vec::with_capacity(self.len() / 2);

        let slice_all = PySliceIndices {
            start: 0,
            stop: self.len() as isize,
            step: 1,
            slicelength: self.len() as isize,
        };
        for (i, str_val) in self
            .get_string_column_range(col_idx, &slice_all)
            .enumerate()
        {
            if !strings_to_remove.contains(&str_val) {
                retained_indices.push(i);
            }
        }

        let table_view = TableView {
            uuid: Uuid::new_v4(),
            desc_uuid: self.desc.uuid,
            index_mapping: IndexMapping::Indices(self.view.uuid, retained_indices),
        };
        let table_view_archived = unsafe { make_mmapped(&table_view) };
        Ok(TableViewMem {
            view: Arc::new(table_view_archived),
            desc: Arc::clone(&self.desc),
            storage: None,
            concat_views: None,
            referenced_view: Some(Box::new(self.clone())),
            live_columns: self.live_columns.clone(),
        })
    }
}

impl TableViewMem {
    /// Run different closures depending on the kind of index mapping
    fn map_index_mapping<'a, SF, CF, IF, RF, O>(
        &'a self,
        col: usize,
        storage_fun: SF,
        concat_fun: CF,
        indices_fun: IF,
        range_fun: RF,
    ) -> O
    where
        SF: FnOnce(&'a ArchivedColumnStorage) -> O,
        CF: FnOnce(&'a [TableViewMem]) -> O,
        IF: FnOnce(&'a TableViewMem, &[u64]) -> O,
        RF: FnOnce(&'a TableViewMem, usize, isize, isize) -> O,
    {
        match &self.view.index_mapping {
            ArchivedIndexMapping::Storage(_storage_uuid) => {
                let storage = self
                    .storage
                    .as_ref()
                    .expect("storage not set when IndexMapping is Storage");
                let col_storage = &storage.columns[col];
                storage_fun(col_storage)
            }
            ArchivedIndexMapping::Concat(_view_uuids) => {
                let (concat_views, _cum_lengths) = self
                    .concat_views
                    .as_ref()
                    .expect("concat_views not set when IndexMapping is Concat");
                concat_fun(concat_views)
            }
            ArchivedIndexMapping::Indices(_view_uuid, indices) => {
                let referenced_view = self
                    .referenced_view
                    .as_ref()
                    .expect("referenced_view not set when IndexMapping is Indices");
                indices_fun(referenced_view, indices)
            }
            ArchivedIndexMapping::Range {
                table_uuid: _,
                start,
                stop,
                step,
            } => {
                let referenced_view = self
                    .referenced_view
                    .as_ref()
                    .expect("referenced_view not set when IndexMapping is Range");
                range_fun(
                    referenced_view,
                    *start as usize,
                    *stop as isize,
                    *step as isize,
                )
            }
        }
    }

    /// Get the values for a column at a given index, assuming the column dtype is f32.
    // AFAICT there's no way to get Rust's type system to let us make this generic over the dtype.
    // We could return untyped NumPy arrays but we want to avoid holding the GIL where possible.
    // Could theoretically use macros. Or, potentially better, make everything operate on byte
    // arrays and push everything to cares about dtype up to the top level. Would make string
    // handling more complicated...
    fn get_f32_column_at_idx(&self, col: usize, idx: usize) -> Box<dyn Iterator<Item = f32> + '_> {
        self.map_index_mapping(
            col,
            |col_storage| {
                let dims_product = self.col_dims_product(col);
                let start = idx * dims_product;
                let end = start + dims_product;
                match col_storage {
                    ArchivedColumnStorage::F32(data) => Box::new(data[start..end].iter().copied())
                        as Box<dyn Iterator<Item = f32> + '_>,
                    _ => panic!("get_f32_column_at_idx called on non-f32 column"),
                }
            },
            |_concat_views: &[TableViewMem]| {
                let (subview, inner_idx) = self.get_subview_and_idx(idx).unwrap();
                subview.get_f32_column_at_idx(col, inner_idx)
            },
            |tgt_tbl, indices| tgt_tbl.get_f32_column_at_idx(col, indices[idx] as usize),
            |tgt_tbl, start, _end, step| {
                tgt_tbl
                    .get_f32_column_at_idx(col, (start as isize + ((idx as isize) * step)) as usize)
            },
        )
    }

    /// Get the values for a column at a given index, assuming the column dtype is i32.
    fn get_i32_column_at_idx(&self, col: usize, idx: usize) -> Box<dyn Iterator<Item = i32> + '_> {
        self.map_index_mapping(
            col,
            |col_storage| {
                let dims_product = self.col_dims_product(col);
                let start = idx * dims_product;
                let end = start + dims_product;
                match col_storage {
                    ArchivedColumnStorage::I32(data) => Box::new(data[start..end].iter().copied())
                        as Box<dyn Iterator<Item = i32> + '_>,
                    _ => panic!("get_i32_column_at_idx called on non-i32 column"),
                }
            },
            |_concat_views: &[TableViewMem]| {
                let (subview, inner_idx) = self.get_subview_and_idx(idx).unwrap();
                subview.get_i32_column_at_idx(col, inner_idx)
            },
            |tgt_tbl, indices| tgt_tbl.get_i32_column_at_idx(col, indices[idx] as usize),
            |tgt_tbl, start, _end, step| {
                tgt_tbl
                    .get_i32_column_at_idx(col, (start as isize + (idx as isize) * step) as usize)
            },
        )
    }

    /// Get the values for a column at a given index, assuming the column dtype is i64.
    fn get_i64_column_at_idx(&self, col: usize, idx: usize) -> Box<dyn Iterator<Item = i64> + '_> {
        self.map_index_mapping(
            col,
            |col_storage| {
                let dims_product = self.col_dims_product(col);
                let start = idx * dims_product;
                let end = start + dims_product;
                match col_storage {
                    ArchivedColumnStorage::I64(data) => Box::new(data[start..end].iter().copied())
                        as Box<dyn Iterator<Item = i64> + '_>,
                    _ => panic!("get_i64_column_at_idx called on non-i64 column"),
                }
            },
            |_concat_views: &[TableViewMem]| {
                let (subview, inner_idx) = self.get_subview_and_idx(idx).unwrap();
                subview.get_i64_column_at_idx(col, inner_idx)
            },
            |tgt_tbl, indices| tgt_tbl.get_i64_column_at_idx(col, indices[idx] as usize),
            |tgt_tbl, start, _end, step| {
                tgt_tbl
                    .get_i64_column_at_idx(col, (start as isize + (idx as isize) * step) as usize)
            },
        )
    }

    /// Get the values for a column at a given index, assuming the column dtype is string.
    fn get_string_column_at_idx(
        &self,
        col: usize,
        idx: usize,
    ) -> Box<dyn Iterator<Item = String> + '_> {
        self.map_index_mapping(
            col,
            |col_storage| {
                let dims_product = self.col_dims_product(col);
                let start = idx * dims_product;
                let end = start + dims_product;
                match col_storage {
                    ArchivedColumnStorage::UString(data) => {
                        Box::new(data[start..end].iter().map(|s| s.to_string()))
                            as Box<dyn Iterator<Item = String> + '_>
                    }
                    _ => panic!("get_string_column_at_idx called on non-string column"),
                }
            },
            |_concat_views: &[TableViewMem]| {
                let (subview, inner_idx) = self.get_subview_and_idx(idx).unwrap();
                subview.get_string_column_at_idx(col, inner_idx)
            },
            |tgt_tbl, indices| tgt_tbl.get_string_column_at_idx(col, indices[idx] as usize),
            |tgt_tbl, start, _end, step| {
                tgt_tbl.get_string_column_at_idx(
                    col,
                    (start as isize + (idx as isize) * step) as usize,
                )
            },
        )
    }

    /// Get the indices in the storage for a range of indices in the view.
    fn get_contiguous_range_storage_indices(
        &self,
        col: usize,
        start: usize,
        stop: usize,
    ) -> (usize, usize) {
        let dims_product = self.col_dims_product(col);
        (start * dims_product, stop * dims_product)
    }

    /// Get the subviews to use and the ranges within them for a range of indices in a concat view.
    fn get_contiguous_range_subviews(
        &self,
        start: usize,
        stop: usize,
    ) -> Vec<(&TableViewMem, usize, usize)> {
        let (start_subview_idx, start_inner_idx) =
            self.get_subview_idx_and_inner_idx(start).unwrap();
        let (end_subview_idx, end_inner_idx) =
            self.get_subview_idx_and_inner_idx(stop - 1).unwrap();
        let end_inner_idx = end_inner_idx + 1;
        let subviews_to_use = &self
            .concat_views
            .as_ref()
            .expect("concat_views not set when IndexMapping is Concat")
            .0[start_subview_idx..=end_subview_idx];
        let mut inner_ranges = Vec::with_capacity(subviews_to_use.len());
        if start_subview_idx == end_subview_idx {
            inner_ranges = vec![(start_inner_idx, end_inner_idx)];
        } else {
            inner_ranges.push((start_inner_idx, subviews_to_use[0].len()));
            for subview in &subviews_to_use[1..subviews_to_use.len() - 1] {
                inner_ranges.push((0, subview.len()));
            }
            inner_ranges.push((0, end_inner_idx));
        }
        inner_ranges
            .into_iter()
            .zip(subviews_to_use)
            .map(|((start, end), subview)| (subview, start, end))
            .collect()
    }

    /// Get a range of a column, assuming the column dtype is f32.
    fn get_f32_column_range(
        &self,
        col: usize,
        slice: &PySliceIndices,
    ) -> Box<dyn Iterator<Item = f32> + '_> {
        if slice.slicelength == 0 {
            return Box::new(std::iter::empty());
        }

        // Fallback if clever strategies don't work
        let fallback = || {
            let indices = PySliceIter::new(slice);
            Box::new(indices.flat_map(move |idx| self.get_f32_column_at_idx(col, idx)))
        };

        if slice.step == 1 {
            // In this case the range is contiguous
            self.map_index_mapping(
                col,
                |col_storage| {
                    let (start_inner_idx, end_inner_idx) = self
                        .get_contiguous_range_storage_indices(
                            col,
                            slice.start as usize,
                            slice.stop as usize,
                        );
                    match col_storage {
                        ArchivedColumnStorage::F32(data) => {
                            Box::new(data[start_inner_idx..end_inner_idx].iter().copied())
                                as Box<dyn Iterator<Item = f32> + '_>
                        }
                        _ => panic!("get_f32_column_range called on non-f32 column"),
                    }
                },
                |_concat_views: &[TableViewMem]| {
                    let inner_ranges = self
                        .get_contiguous_range_subviews(slice.start as usize, slice.stop as usize);
                    Box::new(
                        inner_ranges
                            .into_iter()
                            .flat_map(move |(subview, start, end)| {
                                subview.get_f32_column_range(
                                    col,
                                    &PySliceIndices {
                                        start: start as isize,
                                        stop: end as isize,
                                        step: 1,
                                        slicelength: (end - start) as isize,
                                    },
                                )
                            }),
                    )
                },
                |_tgt_tbl, _indices| fallback(),
                |tgt_tbl, inner_start, _inner_end, inner_step| {
                    if inner_step == 1 {
                        // If the inner range is contiguous then we can just use the same range
                        // with an offset
                        let start = slice.start as usize + inner_start;
                        let end = slice.stop as usize + inner_start;
                        Box::new(tgt_tbl.get_f32_column_range(
                            col,
                            &PySliceIndices {
                                start: start as isize,
                                stop: end as isize,
                                step: 1,
                                slicelength: (end - start) as isize,
                            },
                        ))
                    } else {
                        // Otherwise we have to do a fallback
                        fallback()
                    }
                },
            )
        } else {
            fallback()
        }
    }

    /// Get a range of a column, assuming the column dtype is f32.
    fn get_i32_column_range(
        &self,
        col: usize,
        slice: &PySliceIndices,
    ) -> Box<dyn Iterator<Item = i32> + '_> {
        if slice.slicelength == 0 {
            return Box::new(std::iter::empty());
        }

        // Fallback if clever strategies don't work
        let fallback = || {
            let indices = PySliceIter::new(slice);
            Box::new(indices.flat_map(move |idx| self.get_i32_column_at_idx(col, idx)))
        };

        if slice.step == 1 {
            // In this case the range is contiguous
            self.map_index_mapping(
                col,
                |col_storage| {
                    let (start_inner_idx, end_inner_idx) = self
                        .get_contiguous_range_storage_indices(
                            col,
                            slice.start as usize,
                            slice.stop as usize,
                        );
                    match col_storage {
                        ArchivedColumnStorage::I32(data) => {
                            Box::new(data[start_inner_idx..end_inner_idx].iter().copied())
                                as Box<dyn Iterator<Item = i32> + '_>
                        }
                        _ => panic!("get_i32_column_range called on non-i32 column"),
                    }
                },
                |_concat_views: &[TableViewMem]| {
                    let inner_ranges = self
                        .get_contiguous_range_subviews(slice.start as usize, slice.stop as usize);
                    Box::new(
                        inner_ranges
                            .into_iter()
                            .flat_map(move |(subview, start, end)| {
                                subview.get_i32_column_range(
                                    col,
                                    &PySliceIndices {
                                        start: start as isize,
                                        stop: end as isize,
                                        step: 1,
                                        slicelength: (end - start) as isize,
                                    },
                                )
                            }),
                    )
                },
                |_tgt_tbl, _indices| fallback(),
                |tgt_tbl, inner_start, _inner_end, inner_step| {
                    if inner_step == 1 {
                        // If the inner range is contiguous then we can just use the same range
                        // with an offset
                        let start = slice.start as usize + inner_start;
                        let end = slice.stop as usize + inner_start;
                        Box::new(tgt_tbl.get_i32_column_range(
                            col,
                            &PySliceIndices {
                                start: start as isize,
                                stop: end as isize,
                                step: 1,
                                slicelength: (end - start) as isize,
                            },
                        ))
                    } else {
                        // Otherwise we have to do a fallback
                        fallback()
                    }
                },
            )
        } else {
            fallback()
        }
    }

    /// Get a range of a column, assuming the column dtype is i64.
    fn get_i64_column_range(
        &self,
        col: usize,
        slice: &PySliceIndices,
    ) -> Box<dyn Iterator<Item = i64> + '_> {
        if slice.slicelength == 0 {
            return Box::new(std::iter::empty());
        }

        // Fallback if clever strategies don't work
        let fallback = || {
            let indices = PySliceIter::new(slice);
            Box::new(indices.flat_map(move |idx| self.get_i64_column_at_idx(col, idx)))
        };

        if slice.step == 1 {
            // In this case the range is contiguous
            self.map_index_mapping(
                col,
                |col_storage| {
                    let (start_inner_idx, end_inner_idx) = self
                        .get_contiguous_range_storage_indices(
                            col,
                            slice.start as usize,
                            slice.stop as usize,
                        );
                    match col_storage {
                        ArchivedColumnStorage::I64(data) => {
                            Box::new(data[start_inner_idx..end_inner_idx].iter().copied())
                                as Box<dyn Iterator<Item = i64> + '_>
                        }
                        _ => panic!("get_i64_column_range called on non-i64 column"),
                    }
                },
                |_concat_views: &[TableViewMem]| {
                    let inner_ranges = self
                        .get_contiguous_range_subviews(slice.start as usize, slice.stop as usize);
                    Box::new(
                        inner_ranges
                            .into_iter()
                            .flat_map(move |(subview, start, end)| {
                                subview.get_i64_column_range(
                                    col,
                                    &PySliceIndices {
                                        start: start as isize,
                                        stop: end as isize,
                                        step: 1,
                                        slicelength: (end - start) as isize,
                                    },
                                )
                            }),
                    )
                },
                |_tgt_tbl, _indices| fallback(),
                |tgt_tbl, inner_start, _inner_end, inner_step| {
                    if inner_step == 1 {
                        // If the inner range is contiguous then we can just use the same range
                        // with an offset
                        let start = slice.start as usize + inner_start;
                        let end = slice.stop as usize + inner_start;
                        Box::new(tgt_tbl.get_i64_column_range(
                            col,
                            &PySliceIndices {
                                start: start as isize,
                                stop: end as isize,
                                step: 1,
                                slicelength: (end - start) as isize,
                            },
                        ))
                    } else {
                        // Otherwise we have to do a fallback
                        fallback()
                    }
                },
            )
        } else {
            fallback()
        }
    }

    /// Get a range of a column, assuming the column dtype is UString.
    fn get_string_column_range(
        &self,
        col: usize,
        slice: &PySliceIndices,
    ) -> Box<dyn Iterator<Item = String> + '_> {
        if slice.slicelength == 0 {
            return Box::new(std::iter::empty());
        }

        // Fallback if clever strategies don't work
        let fallback = || {
            let indices = PySliceIter::new(slice);
            Box::new(indices.flat_map(move |idx| self.get_string_column_at_idx(col, idx)))
        };

        if slice.step == 1 {
            // In this case the range is contiguous
            self.map_index_mapping(
                col,
                |col_storage| {
                    let (start_inner_idx, end_inner_idx) = self
                        .get_contiguous_range_storage_indices(
                            col,
                            slice.start as usize,
                            slice.stop as usize,
                        );
                    match col_storage {
                        ArchivedColumnStorage::UString(data) => Box::new(
                            data[start_inner_idx..end_inner_idx]
                                .iter()
                                .map(|s| s.to_string()),
                        )
                            as Box<dyn Iterator<Item = String> + '_>,
                        _ => panic!("get_string_column_range called on non-string column"),
                    }
                },
                |_concat_views: &[TableViewMem]| {
                    let inner_ranges = self
                        .get_contiguous_range_subviews(slice.start as usize, slice.stop as usize);
                    Box::new(
                        inner_ranges
                            .into_iter()
                            .flat_map(move |(subview, start, end)| {
                                subview.get_string_column_range(
                                    col,
                                    &PySliceIndices {
                                        start: start as isize,
                                        stop: end as isize,
                                        step: 1,
                                        slicelength: (end - start) as isize,
                                    },
                                )
                            }),
                    )
                },
                |_tgt_tbl, _indices| fallback(),
                |tgt_tbl, inner_start, _inner_end, inner_step| {
                    if inner_step == 1 {
                        // If the inner range is contiguous then we can just use the same range
                        // with an offset
                        let start = slice.start as usize + inner_start;
                        let end = slice.stop as usize + inner_start;
                        Box::new(tgt_tbl.get_string_column_range(
                            col,
                            &PySliceIndices {
                                start: start as isize,
                                stop: end as isize,
                                step: 1,
                                slicelength: (end - start) as isize,
                            },
                        ))
                    } else {
                        // Otherwise we have to do a fallback
                        fallback()
                    }
                },
            )
        } else {
            fallback()
        }
    }

    /// For views with Concat IndexMappings, find the index of the sub-view and the index within
    /// that sub-view that contain a given index in the outer view.
    fn get_subview_idx_and_inner_idx(&self, idx: usize) -> PyResult<(usize, usize)> {
        match &self.view.index_mapping {
            ArchivedIndexMapping::Concat(_views) => (),
            _ => panic!("get_subview_idx_and_inner_idx called on non-concat view"),
        }
        let (concat_views, cum_lengths) = self
            .concat_views
            .as_ref()
            .expect("concat_views not set when IndexMapping is Concat");
        assert_eq!(concat_views.len(), cum_lengths.len());

        if idx >= *cum_lengths.last().unwrap() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "Index out of bounds",
            ));
        }
        // Find the sub-view by binary search. We want to find the first sub-view whose cumulative
        // length is greater than the index.
        let mut low = 0;
        let mut high = concat_views.len() - 1;
        while low < high {
            let mid = (low + high) / 2;
            if cum_lengths[mid] <= idx {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        assert_eq!(low, high);
        let subview_idx = low;
        let inner_idx = if subview_idx == 0 {
            idx
        } else {
            idx - cum_lengths[subview_idx - 1]
        };

        Ok((subview_idx, inner_idx))
    }

    /// For views with Concat IndexMappings, find the sub-view and the index within that sub-view
    /// that contain a given index in the outer view.
    fn get_subview_and_idx(&self, idx: usize) -> PyResult<(&TableViewMem, usize)> {
        let (subview_idx, inner_idx) = self.get_subview_idx_and_inner_idx(idx)?;
        let subview = &self
            .concat_views
            .as_ref()
            .expect("concat_views not set when IndexMapping is Concat")
            .0[subview_idx];
        Ok((subview, inner_idx))
    }
    /// The product of the dimensions of a column - i.e. the number of elements per row, 5*4 rows
    /// have 20 elements
    fn col_dims_product(&self, col: usize) -> usize {
        self.desc.columns[col].dims.iter().product::<u64>() as usize
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
        if value.shape().is_empty() {
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
    let td_archived = unsafe { make_mmapped(&td) };

    let ts_uuid = Uuid::new_v4();
    let ts = TableStorage {
        uuid: ts_uuid,
        columns: column_storages,
    };
    let ts_archived = unsafe { make_mmapped(&ts) };

    Ok((td_archived, ts_archived))
}

static mut TMPDIR_PATH: Option<PathBuf> = None;

/// Set up the directory to store objects in. Not threadsafe.
unsafe fn setup_tmpdir_path() {
    assert_eq!(TMPDIR_PATH, None);
    match env::var("INFINIDATA_TMPDIR") {
        Ok(val) => {
            let path = Path::new(&val)
                .canonicalize()
                .expect("INFINIDATA_TMPDIR is not a valid path");
            if path.is_dir() || fs::create_dir_all(&path).is_ok() {
                TMPDIR_PATH = Some(path.to_path_buf());
            } else {
                panic!("INFINIDATA_TMPDIR is set but is not a valid path.");
            }
        }
        Err(_) => {
            let cwd = env::current_dir().unwrap();
            let default_path = cwd.join(".infinidata_tmp");
            if default_path.is_dir() || fs::create_dir_all(&default_path).is_ok() {
                TMPDIR_PATH = Some(default_path);
            } else {
                panic!("Could not create default tmpdir path.");
            }
        }
    }
}

trait HasUuid {
    fn get_uuid(&self) -> Uuid;
}

impl HasUuid for TableDesc {
    fn get_uuid(&self) -> Uuid {
        self.uuid
    }
}

impl HasUuid for TableStorage {
    fn get_uuid(&self) -> Uuid {
        self.uuid
    }
}

impl HasUuid for TableView {
    fn get_uuid(&self) -> Uuid {
        self.uuid
    }
}

/// Given an Archive type, write it to disk and return the mmapped struct. Unsafe because it relies
/// on TMPDIR_PATH being set up.
unsafe fn make_mmapped<T>(obj: &T) -> MmapArchived<T>
where
    T: HasUuid + Archive + Serialize<FileSerializer>,
    for<'a> T::Archived: CheckBytes<DefaultValidator<'a>>,
{
    if let Some(tmpdir) = &TMPDIR_PATH {
        let name: PathBuf = format!("{}-{}.bin", type_name::<T>(), obj.get_uuid()).into();
        let path = tmpdir.join(name);
        let file = write_serialize(obj, &path)
            .unwrap_or_else(|err| panic!("writing to tmpdir failed, path was {:?}: {}", path, err));
        load_archived::<T>(file, &path).unwrap_or_else(|err| {
            panic!("loading from tmpdir failed, path was {:?}: {}", path, err)
        })
    } else {
        panic!("TMPDIR_PATH not set up");
    }
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
            // NumPy arrays of strings are packed arrays of UCS4 codepoints with a maximum size.
            // it's not well supported in rust-numpy, and I think the perf difference is probably
            // irrelevant, so we convert to String and use a Vec<String> instead.
            let _str_max_len = dtype_py.itemsize() / 4;
            Ok(DType::UString)
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
        let np = py.import(intern!(py, "numpy")).unwrap();
        let fun = np.getattr(intern!(py, "ascontiguousarray")).unwrap();
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
        DType::UString => {
            let total_len: usize = array.shape().iter().product();
            let arr =
                reshape_pyuntypedarray(py, array, &[total_len]).expect("reshape to flat failed");
            let mut strs = Vec::with_capacity(total_len);
            for str in arr.iter().unwrap() {
                let str = str.unwrap().extract::<String>()?;
                strs.push(str);
            }
            assert_eq!(strs.len(), total_len);
            ColumnStorage::UString(strs)
        }
    };
    Ok(storage)
}

#[pyclass]
struct BatchIter {
    result_queue_recv: Receiver<Arc<Mvar<PreparedBatch>>>,
    desc: Arc<MmapArchived<TableDesc>>,
}

/// A batch of data, with a HashMap from column index to the data for that column, and the number
/// of items in the batch.
type PreparedBatch = (HashMap<usize, ColumnStorage>, usize);

impl BatchIter {
    fn new(
        view: TableViewMem,
        batch_size: usize,
        drop_last_batch: bool,
        threads: u32,
        readahead: u32,
    ) -> Self {
        // This is an iterator that yields batches, fetching them in parallel with readahead.
        // One thread fills a work queue with batches to fetch, creating MVars to put the results
        // in. The MVars are put into a result queue. When Python calls __next__ we try to read
        // from the first MVar in result_queue. If it's empty we block until it's filled. If it's
        // full we pop it and return the batch. If result_queue is empty we block until it's not,
        // or it's closed.

        assert!(threads > 0);
        assert!(readahead > 0);
        assert!(threads <= readahead);

        let (work_tx, work_rx) = crossbeam_channel::unbounded();
        let (result_tx, result_rx) = crossbeam_channel::bounded(readahead as usize);

        // Launch work queue filler thread
        {
            let view = view.clone();
            thread::spawn(move || {
                let mut batch_idx = 0;
                loop {
                    let batch_start = batch_idx * batch_size;
                    let batch_stop = batch_start + batch_size;

                    if batch_stop > view.len() && drop_last_batch {
                        break;
                    }

                    if batch_start >= view.len() {
                        break;
                    }

                    let batch_stop = min(batch_stop, view.len());
                    let slice = PySliceIndices {
                        start: batch_start as isize,
                        stop: batch_stop as isize,
                        step: 1,
                        slicelength: (batch_stop - batch_start) as isize,
                    };

                    let mvar = Arc::new(Mvar::empty());
                    work_tx.send((slice, Arc::clone(&mvar))).unwrap();
                    match result_tx.send(mvar) {
                        Ok(()) => (),
                        Err(_) => break, // result_rx has been dropped, which means the iterator has been dropped
                    }

                    batch_idx += 1;
                }
            });
        }

        // Launch worker threads
        for _ in 0..threads {
            let view = view.clone();
            let work_rx = work_rx.clone();
            let cols = view.live_columns.clone();
            thread::spawn(move || {
                // Iterating over work_rx continues until work_tx is dropped.
                for (slice, mvar) in work_rx {
                    let mut batch = HashMap::with_capacity(cols.len());
                    for col in &cols {
                        let col_storage = match view.desc.columns[*col].dtype {
                            ArchivedDType::F32 => {
                                let col_iter = view.get_f32_column_range(*col, &slice);
                                ColumnStorage::F32(col_iter.collect())
                            }
                            ArchivedDType::I32 => {
                                let col_iter = view.get_i32_column_range(*col, &slice);
                                ColumnStorage::I32(col_iter.collect())
                            }
                            ArchivedDType::I64 => {
                                let col_iter = view.get_i64_column_range(*col, &slice);
                                ColumnStorage::I64(col_iter.collect())
                            }
                            ArchivedDType::UString => {
                                let col_iter = view.get_string_column_range(*col, &slice);
                                ColumnStorage::UString(col_iter.collect())
                            }
                        };
                        batch.insert(*col, col_storage);
                    }
                    mvar.put((batch, slice.slicelength as usize)).unwrap();
                }
            });
        }

        Self {
            result_queue_recv: result_rx,
            desc: view.desc,
        }
    }
}

#[pymethods]
impl BatchIter {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> Option<Py<PyDict>> {
        match self.result_queue_recv.recv() {
            Ok(mvar) => {
                // There is at least one batch currently being processed.
                let (batch, batch_size): (HashMap<usize, ColumnStorage>, usize) =
                    mvar.take().unwrap();
                let dict = PyDict::new(py);
                for (col, storage) in batch {
                    let arr: &PyUntypedArray = match storage {
                        ColumnStorage::F32(data) => {
                            np::PyArray::from_vec(py, data).downcast().unwrap()
                        }
                        ColumnStorage::I32(data) => {
                            np::PyArray::from_vec(py, data).downcast().unwrap()
                        }
                        ColumnStorage::I64(data) => {
                            np::PyArray::from_vec(py, data).downcast().unwrap()
                        }
                        ColumnStorage::UString(data) => {
                            let strings_list = PyList::new(py, data);
                            let np = py.import(intern!(py, "numpy")).unwrap();
                            let fun = np.getattr(intern!(py, "array")).unwrap();
                            fun.call1((strings_list,)).unwrap().downcast().unwrap()
                        }
                    };
                    let col_desc = &self.desc.columns[col];
                    let out_dims = std::iter::once(batch_size)
                        .chain(col_desc.dims.iter().map(|&d| d as usize))
                        .collect::<Vec<usize>>();
                    let arr = reshape_pyuntypedarray(py, arr, &out_dims).unwrap();
                    dict.set_item(col_desc.name.to_string(), arr).unwrap();
                }

                Some(dict.into())
            }
            Err(_) => {
                // Result queue has been dropped, which means the work queue filler thread exited
                // and there are no more batches to fetch.
                None
            }
        }
    }
}

// For some reason there's no Rust interface to reshape on PyUntypedArrays, so we have to go via
// Python
fn reshape_pyuntypedarray<'py>(
    py: Python<'py>,
    array: &'py PyUntypedArray,
    shape: &[usize],
) -> PyResult<&'py PyUntypedArray> {
    let shape: &'py PyTuple = PyTuple::new(py, shape);
    let array = array.call_method1(intern!(py, "reshape"), shape)?;
    Ok(array
        .downcast::<PyUntypedArray>()
        .expect("reshape didn't return an array"))
}

fn concat_pyuntypedarrays<'py>(
    py: Python<'py>,
    arrays: Vec<&'py PyUntypedArray>,
) -> PyResult<&'py PyUntypedArray> {
    let arrays = PyList::new(py, arrays);
    let module = py.import(intern!(py, "numpy")).unwrap();
    let fun = module.getattr(intern!(py, "concatenate")).unwrap();
    let out = fun.call1((arrays,))?;
    Ok(out
        .downcast::<PyUntypedArray>()
        .expect("concatenate didn't return an array"))
}

mod py_slice_iter {
    use super::*;
    pub enum PySliceIter {
        PySliceIter(PySliceIndices),
    }

    impl PySliceIter {
        pub fn new(indices: &PySliceIndices) -> Self {
            PySliceIter::PySliceIter(PySliceIndices {
                start: indices.start,
                stop: indices.stop,
                step: indices.step,
                slicelength: indices.slicelength,
            })
        }
    }

    /// Turn a PySliceIndices into an iterator that yields the indices in the slice
    impl Iterator for PySliceIter {
        type Item = usize;

        fn next(&mut self) -> Option<Self::Item> {
            let PySliceIter::PySliceIter(indices) = self;

            match indices.step.cmp(&0) {
                Ordering::Equal => None,
                Ordering::Greater => {
                    if indices.start >= indices.stop {
                        return None;
                    }
                    let out = indices.start as usize;
                    indices.start += indices.step;
                    indices.slicelength -= 1;
                    Some(out)
                }
                Ordering::Less => {
                    if indices.start <= indices.stop {
                        return None;
                    }
                    let out = indices.start as usize;
                    indices.start += indices.step;
                    indices.slicelength -= 1;
                    Some(out)
                }
            }
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            let PySliceIter::PySliceIter(indices) = self;
            let len = indices.slicelength as usize;
            (len, Some(len))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_py_slice_iter_empty() {
        let indices = PySliceIndices {
            start: 0,
            stop: 0,
            step: 1,
            slicelength: 0,
        };
        let mut iter = PySliceIter::new(&indices);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_py_slice_iter_positive_step() {
        let indices = PySliceIndices {
            start: 0,
            stop: 10,
            step: 2,
            slicelength: 5,
        };
        let mut iter = PySliceIter::new(&indices);
        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), Some(4));
        assert_eq!(iter.next(), Some(6));
        assert_eq!(iter.next(), Some(8));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_py_slice_iter_negative_step() {
        let indices = PySliceIndices {
            start: 10,
            stop: 0,
            step: -2,
            slicelength: 5,
        };
        let mut iter = PySliceIter::new(&indices);
        assert_eq!(iter.next(), Some(10));
        assert_eq!(iter.next(), Some(8));
        assert_eq!(iter.next(), Some(6));
        assert_eq!(iter.next(), Some(4));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), None);
    }
}

use py_slice_iter::*;

type FileSerializer = CompositeSerializer<
    WriteSerializer<BufWriter<File>>,
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
    let writer = BufWriter::with_capacity(4 * 1024 * 1024, file);
    let write_ser = WriteSerializer::new(writer);
    let scratch = FallbackScratch::new(HeapScratch::new(), AllocScratch::new());
    let mut ser = CompositeSerializer::new(write_ser, scratch, SharedSerializeMap::new());
    ser.serialize_value(data).map_or_else(
        |e| Err(std::io::Error::new(std::io::ErrorKind::Other, e)),
        |_| Ok(()),
    )?;
    // the file was moved into the bufwriter, and the bufwriter was moved into write_ser, so we
    // need to get them back out to return the file.
    let writer = ser.into_components().0.into_inner();
    let file = writer.into_inner().unwrap();
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
