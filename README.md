# Infinidata

Infinidata is a Python library for working with arbitrarily large datasets. The only limit... *is
your imagination*. And your disk space. And your virtual address space. And your system RAM for
metadata. But *not* your system RAM for the data itself! Everything gets mmapped.

The API documentation is [here](https://enolan.github.io/infinidata/).

You can iterate over datasets, in batches if you like. You can take subsets with ranges, do
arbitrary permutations, concatenate datasets, and shuffle. All without copying the underlying data.

I wrote this after getting frustrated with Huggingface Datasets' awful performance doing these
things. The main differences are:

- Infinidata doesn't leave cache files lying around without cleaning up after itself. The on-disk
  storage is refcounted and files are deleted once they're dropped. Saving stuff permanently is
  optional.
- Infinidata doesn't leak memory like a sieve when doing lots of manipulations.
- Infinidata is written in Rust and compiled to a native extension module instead of being written
  in regular Python.
- Infinidata's disk format is intentionally unstable. If you store data using Infinidata, things
  may break if you upgrade Infinidata, any of its dependencies, rustc, or change processor
  architectures and try to load it again. It's not an archival format and you'll get no sympathy
  from me if your data becomes useless.
- Infinidata intentionally has way less features: it doesn't download stuff, it doesn't do any
  automatic caching, it doesn't do any automatic type conversion (everything comes out as NumPy
  ndarrays), it doesn't integrate with FAISS, and it doesn't support any fancy data sources like
  s3, pandas, parquet, or arrow.
- Infinidata is missing a lot of functionality it probably *should* have. There's no map or sort,
  and filtering is only implemented for strings.

Usage Notes:

  - If you create lots of `TableViews` you can run into the system limit on memory mappings. It's
    the `vm.max_map_count` sysctl on Linux. Increasing it is harmless, but not generally possible
    inside of Docker containers. So you might not be able to use environments that require your
    software to run inside Docker and won't cooperate about the sysctl.
  - Infinidata deletes all its temporary storage on exit, but if the process crashes, it can't.
    Look in `.infinidata_tmp` in the current working directory for any leftover files.
  - If you need to change the location of the temporary storage, you can set the `INFINIDATA_TMPDIR`
    environment variable. If you're saving things with `TableView.save_to_disk`, it's best if the
    tmpdir and your save location are on the same filesystem so hardlinking will work. Otherwise
    it'll have to copy the data.

Building:

Make a virtualenv, then `pip install maturin`. `maturin build --release` will make an optimized
wheel, and put it in `target/wheels`. `maturin develop` will make a development build and install it
into the current environment. `maturin develop --release` will make an optimized build and install
it into the current environment.

Caveats:

  - Reading from Infinidata into NumPy arrays is not super fast. It individually copies each value
    from the mmapped region into the array. So if you have a high ratio of reads to computation, you
    might have problems. This could be fixed by changing Infinidata's in-RAM format to be
    memcpy'able, but I'm unlikely to get around to that anytime soon.
  - Infinidata only supports float32, int32, int64, and unicode strings. It's not *hard* to add
    more, but doing it requires some ugly copy paste so I've avoided it. This would also be fixed by
    changing the in-RAM format.
  - Modern x86-64 CPUs mostly have 48-bit virtual address spaces. The 64-bit thing is kind of a lie.
    With 48 bits, you can address at most 256 TiB of memory. So your dataset has to be less than
    that. Moderner CPUs have [57-bit virtual address
    spaces](https://en.wikipedia.org/wiki/Intel_5-level_paging), which gets you 128 PiB of
    addressable memory. You have to set some hint flags to get mappings in the upper range, and
    since I don't have more than 256 TiB of storage or one of the fancy newfangled CPUs, I haven't
    done that yet. Patches welcome, I guess.
  - If you have a table with a ton of indirections there's no way to flatten them and get a new
    table backed by contiguous memory. You have to round-trip them through another format (though
    that format can just be NumPy). Having some kind of flattening operation built in would be nice.