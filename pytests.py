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


@pytest.mark.xfail
@pytest.mark.parametrize("tbl_view", ["tbl_view_1", "tbl_view_2", "tbl_view_3"])
def test_array_indexing(tbl_view, request):
    tbl_view, tbl_dict = request.getfixturevalue(tbl_view)
    for idx_array in [
        np.array([0, 5, 11]),
        np.array([5, 11, 20]),
        np.array([11, 20, 30]),
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


@pytest.mark.xfail
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
    new_view = tbl_view.new_view(slice(None, None, -1)) # Equivalent to [::-1]

    # Check that the new view has the correct length
    assert len(new_view) == len(tbl_view)

    for idx in range(len(tbl_view)):
        new_dict = new_view[idx]
        for k in new_dict.keys():
            np.testing.assert_array_equal(
                new_dict[k], tbl_dict[k][-idx-1], strict=True
            )