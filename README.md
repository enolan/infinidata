# Infinidata

Infinidata is a Python libary for working with arbitrarily large datasets. The only limit... *is
your imagination*. And your disk space. And your virtual address space. And your system RAM for
metadata. But not your system RAM for the data itself! Everything gets mmapped.

N.B. **Much of this README is aspirational as of 2024-01-07**, this is a work in progress.

You can iterate over datasets, in batches if you like. You can take subsets with ranges, do
arbitrary permutations, concatenate datasets, and shuffle. All without copying the underlying data.

I wrote this after getting frustrated with Huggingface Datasets' awful performance doing these
things. The main differences are:
- Infinidata doesn't leave cache files lying around without cleaning up after itself. The on-disk
  storage is refcounted and files are deleted once they're dropped. Saving stuff permanently isn't
  even implemented yet.
- Infinidata doesn't leak memory like a sieve when doing lots of manipulations.
- Infinidata is written in Rust and compiled to a native extension module instead of being written
  in regular Python.
- Infinidata's disk format is intentionally unstable. If you store data using Infinidata, things
  may break if you upgrade Infinidata, any of its dependencies, rustc, or change processor
  architectures and try to load it again. It's not an archival format and you'll get no sympathy
  from me if your shit gets fucked.
- Infinidata intentionally has way less features: it doesn't download stuff, it doesn't do any
  automatic caching, it doesn't do any automatic type conversion (everything comes out as NumPy
  ndarrays), it doesn't integrate with FAISS, and it doesn't support any fancy data sources like
  s3, pandas, parquet, or arrow.
- Infinidata is missing a lot of functionality it probably *should* have. There's no map, filter,
  or sort. As above, you can't save anything permanently either.