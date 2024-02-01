"""Python tests for the Infinidata API."""

import numpy as np
import pytest
import uuid

import infinidata


# Fixtures for TableView instances
@pytest.fixture
def tbl_view_1():
    tbl_dict = {
        "foo": np.arange(45 * 16 * 2, dtype=np.float32).reshape((45, 16, 2)),
        "bar": np.arange(45, dtype=np.int32),
        "baz": np.array(["hello"] * 45),
    }
    return infinidata.TableView(tbl_dict), tbl_dict


@pytest.fixture
def tbl_view_2():
    tbl_dict = {
        "alpha": np.random.rand(30, 10).astype(np.float32),
        "beta": np.random.randint(0, 100, size=(30,), dtype=np.int32),
        "gamma": np.array(["world"] * 30),
    }
    return infinidata.TableView(tbl_dict), tbl_dict


@pytest.fixture
def tbl_view_3():
    tbl_dict = {"single_col": np.linspace(0, 1, 50, dtype=np.float32)}
    return infinidata.TableView(tbl_dict), tbl_dict


@pytest.mark.parametrize(
    "tbl_view, expected_length",
    [("tbl_view_1", 45), ("tbl_view_2", 30), ("tbl_view_3", 50)],
)
def test_length(tbl_view, expected_length, request):
    tbl_view = request.getfixturevalue(tbl_view)[0]
    assert len(tbl_view) == expected_length


@pytest.mark.parametrize("tbl_view", ["tbl_view_1", "tbl_view_2", "tbl_view_3"])
def test_uuid(tbl_view, request):
    tbl_view = request.getfixturevalue(tbl_view)[0]
    uuid_str = tbl_view.uuid()
    assert isinstance(uuid_str, str)
    try:
        uuid_obj = uuid.UUID(uuid_str, version=4)
        assert str(uuid_obj) == uuid_str
    except ValueError:
        pytest.fail("uuid() method did not return a valid UUID")


@pytest.mark.parametrize("tbl_view", ["tbl_view_1", "tbl_view_2", "tbl_view_3"])
def test_single_indexing(tbl_view, request):
    tbl_view, tbl_dict = request.getfixturevalue(tbl_view)
    for idx in [0, 5, 11]:
        view_dict = tbl_view[idx]
        for key in tbl_dict.keys():
            np.testing.assert_array_equal(
                view_dict[key], tbl_dict[key][idx], strict=True
            )


@pytest.mark.parametrize("tbl_view", ["tbl_view_1", "tbl_view_2", "tbl_view_3"])
def test_simple_slicing(tbl_view, request):
    tbl_view, tbl_dict = request.getfixturevalue(tbl_view)
    for idx_slice in [slice(0, 5), slice(5, 11), slice(11, 20)]:
        view_dict = tbl_view[idx_slice]
        for key in tbl_dict.keys():
            np.testing.assert_array_equal(
                view_dict[key], tbl_dict[key][idx_slice], strict=True
            )


@pytest.mark.parametrize("tbl_view", ["tbl_view_1", "tbl_view_2", "tbl_view_3"])
def test_slicing_with_step(tbl_view, request):
    tbl_view, tbl_dict = request.getfixturevalue(tbl_view)
    for idx_slice in [slice(0, 5, 2), slice(5, 11, 3), slice(11, 20, 4)]:
        view_dict = tbl_view[idx_slice]
        for key in tbl_dict.keys():
            np.testing.assert_array_equal(
                view_dict[key], tbl_dict[key][idx_slice], strict=True
            )


@pytest.mark.parametrize("tbl_view", ["tbl_view_1", "tbl_view_2", "tbl_view_3"])
def test_slicing_with_negative_step(tbl_view, request):
    tbl_view, tbl_dict = request.getfixturevalue(tbl_view)
    for idx_slice in [slice(5, 0, -1), slice(11, 5, -2), slice(20, 11, -3)]:
        view_dict = tbl_view[idx_slice]
        for key in tbl_dict.keys():
            np.testing.assert_array_equal(
                view_dict[key], tbl_dict[key][idx_slice], strict=True
            )


@pytest.mark.parametrize("tbl_view", ["tbl_view_1", "tbl_view_2", "tbl_view_3"])
def test_slicing_with_negative_start_and_stop(tbl_view, request):
    tbl_view, tbl_dict = request.getfixturevalue(tbl_view)
    for idx_slice in [slice(-5, -1), slice(-11, -5), slice(-20, -11)]:
        view_dict = tbl_view[idx_slice]
        for key in tbl_dict.keys():
            np.testing.assert_array_equal(
                view_dict[key], tbl_dict[key][idx_slice], strict=True
            )


@pytest.mark.parametrize("tbl_view", ["tbl_view_1", "tbl_view_2", "tbl_view_3"])
def test_slicing_with_negative_start_stop_and_step(tbl_view, request):
    tbl_view, tbl_dict = request.getfixturevalue(tbl_view)
    for idx_slice in [slice(-1, -5, -1), slice(-5, -11, -2), slice(-11, -20, -3)]:
        view_dict = tbl_view[idx_slice]
        for key in tbl_dict.keys():
            np.testing.assert_array_equal(
                view_dict[key], tbl_dict[key][idx_slice], strict=True
            )


@pytest.mark.parametrize("tbl_view", ["tbl_view_1", "tbl_view_2", "tbl_view_3"])
def test_array_indexing(tbl_view, request):
    tbl_view, tbl_dict = request.getfixturevalue(tbl_view)
    for idx_array in [
        np.array([0, 5, 11]),
        np.array([5, 11, 20]),
        np.array([25, 20, 20, 29]),
        np.arange(10)[::-1],
    ]:
        view_dict = tbl_view[idx_array]
        for key in tbl_dict.keys():
            np.testing.assert_array_equal(
                view_dict[key], tbl_dict[key][idx_array], strict=True
            )


# A set of concatenatable TableViews for testing the concat() method
@pytest.fixture
def concatable_tbl_view_1():
    tbl_dict = {
        "alice": np.random.rand(10, 10).astype(np.float32),
        "bob": np.random.randint(0, 100, size=(10,), dtype=np.int32),
        "carol": np.arange(10 * 2 * 4 * 3, dtype=np.int64)[::-1].reshape((10, 2, 4, 3)),
    }
    return infinidata.TableView(tbl_dict), tbl_dict


@pytest.fixture
def concatable_tbl_view_2():
    tbl_dict = {
        "alice": np.linspace(0, 1, 300, dtype=np.float32).reshape((30, 10)),
        "bob": np.arange(30, dtype=np.int32),
        "carol": (np.arange(30 * 2 * 4 * 3, dtype=np.int64) * 22).reshape(
            (30, 2, 4, 3)
        ),
    }
    return infinidata.TableView(tbl_dict), tbl_dict


@pytest.fixture
def concatable_tbl_view_3():
    tbl_dict = {
        "alice": np.random.randn(4, 10).astype(np.float32),
        "bob": np.ones((4,), dtype=np.int32) * 420,
        "carol": np.arange(4 * 2 * 4 * 3, dtype=np.int64)[::-1].reshape((4, 2, 4, 3)),
    }
    return infinidata.TableView(tbl_dict), tbl_dict


concatable_combos = [
    ["concatable_tbl_view_1", "concatable_tbl_view_2"],
    ["concatable_tbl_view_2", "concatable_tbl_view_3"],
    ["concatable_tbl_view_1", "concatable_tbl_view_3"],
    ["concatable_tbl_view_1", "concatable_tbl_view_2", "concatable_tbl_view_3"],
]


@pytest.mark.parametrize("concatable_tbl_views", concatable_combos)
def test_concat_len(concatable_tbl_views, request):
    tbl_views = [request.getfixturevalue(tbl_view) for tbl_view in concatable_tbl_views]
    tbl_views = [tbl_view[0] for tbl_view in tbl_views]
    concat_view = infinidata.TableView.concat(tbl_views)

    # Check that the concatenated view has the correct length
    expected_length = sum([len(tbl_view) for tbl_view in tbl_views])
    assert len(concat_view) == expected_length


@pytest.mark.parametrize("concatable_tbl_views", concatable_combos)
def test_concat_single_indexing(concatable_tbl_views, request):
    tbl_views_and_dicts = [
        request.getfixturevalue(tbl_view) for tbl_view in concatable_tbl_views
    ]
    tbl_views = [tbl_view[0] for tbl_view in tbl_views_and_dicts]
    tbl_dicts = [tbl_view[1] for tbl_view in tbl_views_and_dicts]
    concat_view = infinidata.TableView.concat(tbl_views)

    # Check that the concatenated view has the correct data
    start_idx = 0
    for inner_view in tbl_views:
        for idx in range(len(inner_view)):
            concat_dict = concat_view[start_idx + idx]
            inner_view_dict = inner_view[idx]
            assert list(concat_dict.keys()) == list(inner_view_dict.keys())
            for key in tbl_dicts[0].keys():
                np.testing.assert_array_equal(
                    concat_dict[key], inner_view_dict[key], strict=True
                )
        start_idx += len(inner_view)


@pytest.mark.parametrize("concatable_tbl_views", concatable_combos)
def test_concat_slice_all(concatable_tbl_views, request):
    tbl_views = [request.getfixturevalue(tbl_view) for tbl_view in concatable_tbl_views]
    tbl_views = [tbl_view[0] for tbl_view in tbl_views]
    concat_view = infinidata.TableView.concat(tbl_views)

    concat_dict_all = concat_view[:]

    start_idx = 0
    for inner_view in tbl_views:
        for idx in range(len(inner_view)):
            for k in concat_dict_all.keys():
                np.testing.assert_array_equal(
                    concat_dict_all[k][start_idx + idx], inner_view[idx][k], strict=True
                )
        start_idx += len(inner_view)


@pytest.mark.parametrize("concatable_tbl_views", concatable_combos)
def test_concat_slice_inside(concatable_tbl_views, request):
    tbl_views = [request.getfixturevalue(tbl_view) for tbl_view in concatable_tbl_views]
    tbl_views = [tbl_view[0] for tbl_view in tbl_views]
    concat_view = infinidata.TableView.concat(tbl_views)

    start_idx = 0
    for inner_view in tbl_views:
        for idx_slice in [
            slice(start_idx, start_idx + len(inner_view)),
            slice(start_idx + len(inner_view) // 2, start_idx + len(inner_view)),
            slice(start_idx, start_idx + len(inner_view) // 2),
        ]:
            concat_dict = concat_view[idx_slice]
            inner_slice = slice(idx_slice.start - start_idx, idx_slice.stop - start_idx)
            inner_dict = inner_view[inner_slice]
            for k in concat_dict.keys():
                np.testing.assert_array_equal(
                    concat_dict[k], inner_dict[k], strict=True
                )
        start_idx += len(inner_view)


@pytest.mark.parametrize("concatable_tbl_views", concatable_combos)
def test_concat_slice_across(concatable_tbl_views, request):
    tbl_views = [request.getfixturevalue(tbl_view) for tbl_view in concatable_tbl_views]
    tbl_views = [tbl_view[0] for tbl_view in tbl_views]
    concat_view = infinidata.TableView.concat(tbl_views)

    # Test slicing across the boundary between two concatenated views
    start_idx = 0
    for view_idx in range(len(tbl_views) - 1):
        idx_slice = slice(
            start_idx + len(tbl_views[view_idx]) - 1,
            start_idx + len(tbl_views[view_idx]) + 1,
        )
        concat_dict = concat_view[idx_slice]
        last_dict = tbl_views[view_idx][len(tbl_views[view_idx]) - 1]
        first_dict = tbl_views[view_idx + 1][0]
        for k in concat_dict.keys():
            np.testing.assert_array_equal(concat_dict[k][0], last_dict[k], strict=True)
            np.testing.assert_array_equal(concat_dict[k][1], first_dict[k], strict=True)
        start_idx += len(tbl_views[view_idx])


@pytest.mark.parametrize("concatable_tbl_views", concatable_combos)
def test_concat_array_indexing(concatable_tbl_views, request):
    tbl_views = [request.getfixturevalue(tbl_view) for tbl_view in concatable_tbl_views]
    tbl_views = [tbl_view[0] for tbl_view in tbl_views]
    concat_view = infinidata.TableView.concat(tbl_views)

    n_samples = len(concat_view)

    cum_lens = np.cumsum([len(tbl_view) for tbl_view in tbl_views])
    outer_indices = []
    inner_indices = []

    # Generate a set of random indices, tracking both the index into the outer view and the index
    # of and index into the inner view
    for i in range(n_samples):
        inner_view = np.random.randint(0, len(tbl_views))
        inner_idx = np.random.randint(0, len(tbl_views[inner_view]))
        outer_idx = cum_lens[inner_view] - len(tbl_views[inner_view]) + inner_idx
        outer_indices.append(outer_idx)
        inner_indices.append((inner_view, inner_idx))

    outer_indices = np.array(outer_indices)
    concat_array_dict = concat_view[outer_indices]
    for i in range(n_samples):
        inner_view, inner_idx = inner_indices[i]
        inner_dict = tbl_views[inner_view][inner_idx]
        for k in concat_array_dict.keys():
            np.testing.assert_array_equal(
                concat_array_dict[k][i], inner_dict[k], strict=True
            )


@pytest.mark.parametrize("tbl_view", ["tbl_view_1", "tbl_view_2", "tbl_view_3"])
def test_new_view_array_indexing(tbl_view, request):
    tbl_view, tbl_dict = request.getfixturevalue(tbl_view)
    n_samples = len(tbl_view) // 2

    # Generate a set of random indices
    indices = np.random.randint(0, len(tbl_view), size=(n_samples,))

    # Generate a new view
    new_view = tbl_view.new_view(indices)

    # Check that the new view has the correct length
    assert len(new_view) == n_samples

    # Check that the new view has the correct data
    for i in range(n_samples):
        new_dict = new_view[i]
        for k in new_dict.keys():
            np.testing.assert_array_equal(
                new_dict[k], tbl_dict[k][indices[i]], strict=True
            )


@pytest.mark.parametrize("tbl_view", ["tbl_view_1", "tbl_view_2", "tbl_view_3"])
def test_new_view_slice_all(tbl_view, request):
    tbl_view, tbl_dict = request.getfixturevalue(tbl_view)

    # Generate a new view
    new_view = tbl_view.new_view(slice(None))
    new_view_dict = new_view[:]

    # Check that the new view has the correct length
    assert len(new_view) == len(tbl_view)

    # Check that the new view has the correct data
    for k in new_view_dict.keys():
        np.testing.assert_array_equal(new_view_dict[k], tbl_dict[k], strict=True)


@pytest.mark.parametrize("tbl_view", ["tbl_view_1", "tbl_view_2", "tbl_view_3"])
def test_new_view_slice_contiguous(tbl_view, request):
    tbl_view, tbl_dict = request.getfixturevalue(tbl_view)

    # Generate a new view
    new_view = tbl_view.new_view(slice(5, 10))
    new_view_dict = new_view[:]

    # Check that the new view has the correct length
    assert len(new_view) == 5

    # Check that the new view has the correct data
    for k in new_view_dict.keys():
        np.testing.assert_array_equal(new_view_dict[k], tbl_dict[k][5:10], strict=True)


@pytest.mark.parametrize("tbl_view", ["tbl_view_1", "tbl_view_2", "tbl_view_3"])
def test_new_view_slice_noncontiguous(tbl_view, request):
    tbl_view, tbl_dict = request.getfixturevalue(tbl_view)

    # Generate a new view
    new_view = tbl_view.new_view(slice(3, 15, 3))
    new_view_dict = new_view[:]

    # Check that the new view has the correct length
    assert len(new_view) == 4

    # Check that the new view has the correct data
    for k in new_view_dict.keys():
        np.testing.assert_array_equal(
            new_view_dict[k], tbl_dict[k][3:15:3], strict=True
        )


@pytest.mark.parametrize("tbl_view", ["tbl_view_1", "tbl_view_2", "tbl_view_3"])
def test_new_view_slice_reverse(tbl_view, request):
    tbl_view, tbl_dict = request.getfixturevalue(tbl_view)

    # Generate a new view
    new_view = tbl_view.new_view(slice(None, None, -1))  # Equivalent to [::-1]

    # Check that the new view has the correct length
    assert len(new_view) == len(tbl_view)

    for idx in range(len(tbl_view)):
        new_dict = new_view[idx]
        for k in new_dict.keys():
            np.testing.assert_array_equal(
                new_dict[k], tbl_dict[k][-idx - 1], strict=True
            )


@pytest.mark.parametrize("tbl_view", ["tbl_view_1", "tbl_view_2", "tbl_view_3"])
def test_shuffle(tbl_view, request):
    tbl_view, tbl_dict = request.getfixturevalue(tbl_view)

    # Generate a shuffled view
    shuffled_view = tbl_view.shuffle()

    # Check that the shuffled view has the correct length
    assert len(shuffled_view) == len(tbl_view)

    # Check that the shuffled view has the correct data
    found_rows = np.zeros(len(tbl_view), dtype=np.int32)
    for i in range(len(tbl_view)):
        shuffled_dict = shuffled_view[i]
        for j in range(len(tbl_view)):
            if all(
                np.array_equal(shuffled_dict[k], tbl_dict[k][j])
                for k in shuffled_dict.keys()
            ):
                found_rows[j] += 1
                break
    assert np.all(found_rows == 1)


@pytest.mark.parametrize("tbl_view", ["tbl_view_1", "tbl_view_2", "tbl_view_3"])
def test_batch_iter_drop_last(tbl_view, request):
    tbl_view, tbl_dict = request.getfixturevalue(tbl_view)

    batch_cnt = 0
    for i, batch in enumerate(tbl_view.batch_iter(5, drop_last_batch=True)):
        batch_cnt += 1
        for k in batch.keys():
            assert len(batch[k]) == 5
            np.testing.assert_array_equal(
                batch[k], tbl_dict[k][i * 5 : (i + 1) * 5], strict=True
            )

    assert batch_cnt == len(tbl_view) // 5


@pytest.mark.parametrize("tbl_view", ["tbl_view_1", "tbl_view_2", "tbl_view_3"])
def test_batch_iter_no_drop_last(tbl_view, request):
    tbl_view, tbl_dict = request.getfixturevalue(tbl_view)

    batch_cnt = 0
    for i, batch in enumerate(tbl_view.batch_iter(5, drop_last_batch=False)):
        batch_cnt += 1
        for k in batch.keys():
            if i == len(tbl_view) // 5:
                assert len(batch[k]) == len(tbl_view) % 5
            else:
                assert len(batch[k]) == 5
            np.testing.assert_array_equal(
                batch[k], tbl_dict[k][i * 5 : (i + 1) * 5], strict=True
            )

    assert batch_cnt == len(tbl_view) // 5 + (1 if len(tbl_view) % 5 != 0 else 0)


@pytest.mark.parametrize(
    "threads,readahead", [(1, 0), (1, 2), (1, 10), (2, 2), (2, 10), (8, 8)]
)
def test_batch_iter_threads_no_drop_last(threads, readahead, request):
    tbl_view, tbl_dict = request.getfixturevalue("tbl_view_1")

    batch_cnt = 0
    for i, batch in enumerate(
        tbl_view.batch_iter(
            7, drop_last_batch=False, threads=threads, readahead=readahead
        )
    ):
        batch_cnt += 1
        for k in batch.keys():
            if i == len(tbl_view) // 7:
                assert len(batch[k]) == len(tbl_view) % 7
            else:
                assert len(batch[k]) == 7
            np.testing.assert_array_equal(
                batch[k], tbl_dict[k][i * 7 : (i + 1) * 7], strict=True
            )

    assert batch_cnt == len(tbl_view) // 7 + (1 if len(tbl_view) % 7 != 0 else 0)


@pytest.mark.parametrize(
    "threads,readahead", [(1, 0), (1, 2), (1, 10), (2, 2), (2, 10), (8, 8)]
)
def test_batch_iter_threads_drop_last(threads, readahead, request):
    tbl_view, tbl_dict = request.getfixturevalue("tbl_view_1")

    batch_cnt = 0
    for i, batch in enumerate(
        tbl_view.batch_iter(
            7, drop_last_batch=True, threads=threads, readahead=readahead
        )
    ):
        batch_cnt += 1
        for k in batch.keys():
            assert len(batch[k]) == 7
            np.testing.assert_array_equal(
                batch[k], tbl_dict[k][i * 7 : (i + 1) * 7], strict=True
            )

    assert batch_cnt == len(tbl_view) // 7


@pytest.mark.parametrize("concatable_tbl_views", concatable_combos)
def test_concat_batch_iter_concat_no_drop_last(concatable_tbl_views, request):
    tbl_views = [request.getfixturevalue(tbl_view) for tbl_view in concatable_tbl_views]
    tbl_views = [tbl_view[0] for tbl_view in tbl_views]
    total_rows = sum([len(tbl_view) for tbl_view in tbl_views])

    batches = list(
        infinidata.TableView.batch_iter_concat(
            tbl_views, batch_size=5, drop_last_batch=False, threads=1, readahead=0
        )
    )
    get_batch_len = lambda batch: len(batch[list(batch.keys())[0]])
    batch_lens = [get_batch_len(batch) for batch in batches]

    # Check the batch lengths are correct
    assert len(batches) == total_rows // 5 + (1 if total_rows % 5 != 0 else 0)
    assert all([batch_len == 5 for batch_len in batch_lens[:-1]])
    assert batch_lens[-1] == (total_rows % 5 if total_rows % 5 != 0 else 5)
    assert sum(batch_lens) == total_rows

    # Check the batch data is correct
    dicts = [tbl_view[:] for tbl_view in tbl_views]
    dict_all = {
        k: np.concatenate([d[k] for d in dicts], axis=0) for k in dicts[0].keys()
    }
    for i, batch in enumerate(batches):
        for k in batch.keys():
            np.testing.assert_array_equal(
                batch[k], dict_all[k][i * 5 : (i + 1) * 5], strict=True
            )


def test_concat_batch_iter_tons_of_threads():
    vals = np.arange(100_000)
    tbl_dicts = [{"vals": vals[i : i + 100]} for i in range(0, 100_000, 100)]
    tbl_views = [infinidata.TableView(tbl_dict) for tbl_dict in tbl_dicts]

    batches = list(infinidata.TableView.batch_iter_concat(tbl_views, batch_size=33, drop_last_batch=False, threads=3000, readahead=10_000))
    get_batch_len = lambda batch: len(batch[list(batch.keys())[0]])

    # Check the batch lengths are correct
    assert len(batches) == 3031
    assert all([get_batch_len(batch) == 33 for batch in batches[:-1]])
    assert get_batch_len(batches[-1]) == 10

    # Concatenate the batches
    batch_arrs = [batch["vals"] for batch in batches]
    batch_arr = np.concatenate(batch_arrs, axis=0)

    assert batch_arr.shape == (100_000,)
    np.testing.assert_array_equal(batch_arr, vals, strict=True)


@pytest.mark.parametrize("tbl_view", ["tbl_view_1", "tbl_view_2", "tbl_view_3"])
def test_select_columns(tbl_view, request):
    tbl_view, tbl_dict = request.getfixturevalue(tbl_view)

    # Select a subset of columns
    selected_cols = set()

    for i, k in enumerate(tbl_dict.keys()):
        if i % 2 == 0:
            selected_cols.add(k)
    selected_view = tbl_view.select_columns(selected_cols)

    # Check that the selected view has the correct length
    assert len(selected_view) == len(tbl_view)

    # Check that the selected view has the correct data
    for i in range(len(tbl_view)):
        selected_dict = selected_view[i]
        assert set(selected_dict.keys()) == selected_cols
        for k in selected_dict.keys():
            np.testing.assert_array_equal(selected_dict[k], tbl_dict[k][i], strict=True)


def test_remove_matching_strings():
    tbl_dict = {
        "foo": np.array([str(i) for i in range(100)]),
        "bar": np.arange(100)[::-1],
    }
    tbl_view = infinidata.TableView(tbl_dict)

    blacklist_set = {"foo", "3", "11", "40"}
    tbl_view_filtered = tbl_view.remove_matching_strings("foo", blacklist_set)

    assert len(tbl_view_filtered) == len(tbl_view) - 3

    filtered_dict = tbl_view_filtered[:]

    for i in range(len(tbl_view_filtered)):
        # Check that this row does not have a blacklisted string in the foo column
        assert filtered_dict["foo"][i] not in blacklist_set

        # Check that this row matches the original table by finding the matching idx in the
        # original table.
        matching_idx = i
        while True:
            matches = True
            for k in filtered_dict.keys():
                if filtered_dict[k][i] != tbl_dict[k][matching_idx]:
                    matches = False
                    break
            if matches:
                break
            matching_idx += 1
            assert matching_idx < len(
                tbl_view
            ), f"Couldn't find a matching row for {i} in the original table"


@pytest.mark.parametrize("tbl_view", ["tbl_view_1", "tbl_view_2", "tbl_view_3"])
def test_simple_save_load(tbl_view, request, tmp_path):
    tbl_view, tbl_dict = request.getfixturevalue(tbl_view)

    # Save the table
    tbl_view.save_to_disk(tmp_path, "test_table")

    # Load the table
    loaded_view = infinidata.TableView.load_from_disk(tmp_path, "test_table")

    # Check that the loaded table has the correct length
    assert len(loaded_view) == len(tbl_view)

    # Check that the loaded table has the correct data
    for i in range(len(tbl_view)):
        loaded_dict = loaded_view[i]
        for k in loaded_dict.keys():
            np.testing.assert_array_equal(loaded_dict[k], tbl_dict[k][i], strict=True)


@pytest.mark.parametrize("concatable_tbl_views", concatable_combos)
def test_concat_save_load(concatable_tbl_views, request, tmp_path):
    tbl_views_and_dicts = [
        request.getfixturevalue(tbl_view) for tbl_view in concatable_tbl_views
    ]
    tbl_views = [tbl_view[0] for tbl_view in tbl_views_and_dicts]
    tbl_dicts = [tbl_view[1] for tbl_view in tbl_views_and_dicts]

    # Concatenate the tables
    concat_view = infinidata.TableView.concat(tbl_views)

    # Save the table
    concat_view.save_to_disk(tmp_path, "test_table")

    # Load the table
    loaded_view = infinidata.TableView.load_from_disk(tmp_path, "test_table")

    # Check that the loaded table has the correct length
    assert len(loaded_view) == len(concat_view)

    # Check that the loaded table has the correct data
    start_idx = 0
    for inner_view in tbl_views:
        for i in range(len(inner_view)):
            loaded_dict = loaded_view[start_idx + i]
            inner_dict = inner_view[i]
            assert list(loaded_dict.keys()) == list(inner_dict.keys())
            for k in loaded_dict.keys():
                np.testing.assert_array_equal(
                    loaded_dict[k], inner_dict[k], strict=True
                )
        start_idx += len(inner_view)


@pytest.mark.parametrize("tbl_view", ["tbl_view_1", "tbl_view_2", "tbl_view_3"])
def test_save_load_array_indexing(tbl_view, request, tmp_path):
    tbl_view, tbl_dict = request.getfixturevalue(tbl_view)
    tbl_view = tbl_view.new_view(np.arange(len(tbl_view))[::-1])

    # Save the table
    tbl_view.save_to_disk(tmp_path, "test_table")

    # Load the table
    loaded_view = infinidata.TableView.load_from_disk(tmp_path, "test_table")

    # Check that the loaded table has the correct length
    assert len(loaded_view) == len(tbl_view)

    # Check that the loaded table has the correct data
    for i in range(len(tbl_view)):
        loaded_dict = loaded_view[i]
        for k in loaded_dict.keys():
            np.testing.assert_array_equal(
                loaded_dict[k], tbl_dict[k][len(tbl_view) - i - 1], strict=True
            )
