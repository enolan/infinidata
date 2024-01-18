use core::fmt::Debug;
use memmap::Mmap;
use numpy as np;
use numpy::array::*;
use numpy::PyUntypedArray;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::*;
use rkyv::ser::serializers::{
    AllocScratch, CompositeSerializer, FallbackScratch, HeapScratch, SharedSerializeMap,
    WriteSerializer,
};
use rkyv::ser::Serializer;
use rkyv::validation::validators::{check_archived_root, DefaultValidator};
use rkyv::{Archive, CheckBytes, Deserialize, Serialize};
use std::cmp::Ordering;
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
        start: usize,
        end: usize,
        step: usize,
    },
}

/// A TableView along with the associated in-RAM metadata. This is mostly a collection of Arcs,
/// along with a little runtime data.
#[pyclass(name = "TableView")]
#[derive(Clone, Debug)]
struct TableViewMem {
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
        let path: PathBuf = format!("TableView-{}.bin", view.uuid).into();
        let file = write_serialize(&view, &path).unwrap();
        let view_archived = load_archived::<TableView>(file, &path).unwrap();
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
                let mut arr: &PyUntypedArray;
                match col_desc.dtype {
                    ArchivedDType::F32 => {
                        let iter = self.get_f32_column_at_idx(col, index);
                        arr = np::PyArray::from_iter(py, iter).downcast().unwrap();
                    }
                    ArchivedDType::I32 => {
                        let iter = self.get_i32_column_at_idx(col, index);
                        arr = np::PyArray::from_iter(py, iter).downcast().unwrap();
                    }
                    ArchivedDType::I64 => {
                        let iter = self.get_i64_column_at_idx(col, index);
                        arr = np::PyArray::from_iter(py, iter).downcast().unwrap();
                    }
                    ArchivedDType::UString => {
                        let iter = self.get_string_column_at_idx(col, index);
                        let strings = iter.collect::<Vec<String>>();
                        let string_list = PyList::new(py, strings);
                        let np = py.import(intern!(py, "numpy")).unwrap();
                        let fun = np.getattr(intern!(py, "array")).unwrap();
                        arr = fun.call1((string_list,)).unwrap().downcast().unwrap();
                    }
                }
                let dims: &[usize] = &col_desc
                    .dims
                    .iter()
                    .map(|&d| d as usize)
                    .collect::<Vec<usize>>();
                if !dims.is_empty() {
                    arr = reshape_pyuntypedarray(py, arr, dims).unwrap();
                }
                out.set_item(col_name, arr).unwrap();
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
            ArchivedIndexMapping::Indices(_view_uuid, _indices) => {
                todo!("len for direct indices mapping")
            }
            ArchivedIndexMapping::Range {
                table_uuid: _,
                start: _,
                end: _,
                step: _,
            } => todo!("range len"),
        }
    }

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

        let path: PathBuf = format!("TableView-{}.bin", table_view.uuid).into();
        let file = write_serialize(&table_view, &path).unwrap();
        let table_view_archived = load_archived::<TableView>(file, &path).unwrap();
        Ok(TableViewMem {
            view: Arc::new(table_view_archived),
            desc: Arc::clone(&views[0].extract::<PyRef<Self>>()?.desc),
            storage: None,
            concat_views: Some((concat_views, cum_lengths)),
            referenced_view: None,
            live_columns: (0..all_desc.columns.len()).collect(),
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
        RF: FnOnce(&'a TableViewMem, usize, usize, usize) -> O,
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
                end,
                step,
            } => {
                let referenced_view = self
                    .referenced_view
                    .as_ref()
                    .expect("referenced_view not set when IndexMapping is Range");
                range_fun(
                    referenced_view,
                    *start as usize,
                    *end as usize,
                    *step as usize,
                )
            }
        }
    }

    /// Get the values for a column at a given index, assuming the column dtype is f32.
    // AFAICT there's no way to get Rust's type system to let us make this generic over the dtype.
    // We could return untyped NumPy arrays but we want to avoid holding the GIL where possible.
    // could theoretically use macros.
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
            |_tgt_tbl, _indices| todo!("get_f32_column_at_idx for direct indices mapping"),
            |_tgt_tbl, _start, _end, _step| todo!("get_f32_column_at_idx for Range"),
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
            |_tgt_tbl, _indices| todo!("get_i32_column_at_idx for direct indices mapping"),
            |_tgt_tbl, _start, _end, _step| todo!("get_i32_column_at_idx for Range"),
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
            |_tgt_tbl, _indices| todo!("get_i64_column_at_idx for direct indices mapping"),
            |_tgt_tbl, _start, _end, _step| todo!("get_i64_column_at_idx for Range"),
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
            |_tgt_tbl, _indices| todo!("get_string_column_at_idx for direct indices mapping"),
            |_tgt_tbl, _start, _end, _step| todo!("get_string_column_at_idx for Range"),
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
                |_tgt_tbl, _indices| todo!("get_f32_column_range for direct indices mapping"),
                |_tgt_tbl, _start, _end, _step| todo!("get_f32_column_range for Range"),
            )
        } else {
            let indices = PySliceIter::new(slice);
            Box::new(indices.flat_map(move |idx| self.get_f32_column_at_idx(col, idx)))
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
                |_tgt_tbl, _indices| todo!("get_i32_column_range for direct indices mapping"),
                |_tgt_tbl, _start, _end, _step| todo!("get_i32_column_range for Range"),
            )
        } else {
            let indices = PySliceIter::new(slice);
            Box::new(indices.flat_map(move |idx| self.get_i32_column_at_idx(col, idx)))
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
                |_tgt_tbl, _indices| todo!("get_i64_column_range for direct indices mapping"),
                |_tgt_tbl, _start, _end, _step| todo!("get_i64_column_range for Range"),
            )
        } else {
            let indices = PySliceIter::new(slice);
            Box::new(indices.flat_map(move |idx| self.get_i64_column_at_idx(col, idx)))
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
                |_tgt_tbl, _indices| todo!("get_string_column_range for direct indices mapping"),
                |_tgt_tbl, _start, _end, _step| todo!("get_string_column_range for Range"),
            )
        } else {
            let indices = PySliceIter::new(slice);
            Box::new(indices.flat_map(move |idx| self.get_string_column_at_idx(col, idx)))
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
