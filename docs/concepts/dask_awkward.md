# Dask-Awkward

## Dask

The [dask](https://docs.dask.org/en/stable/) ecosystem is a series of libraries that wrap an existing
python array-based library such as numpy, pandas or awkward. These wrapping delay the execution of a
given operation until dask performs a computation of the whole object. In essence, you defined the
operations that you want to perform and at the very end of the code you call `dask.compute()` which
actually crunshes the numbers. The benefit from this approach is that the overhead required to perform
that computation is know to dask, who can regulate how much resources to use at each point.

Dask is on the one hand a toolbox for build dask-libraries such as dask-pandas DataFrames or dask-numpy
Arrays (called Dask-Dataframes and Dask-Arrays respectively). There are three key steps required to
used dask in your code:

 1. Open data with a dask delayed reader. This means that the data is not actually read, but merely
    inspected and organized into partitions. These partitions determine how dask will process the
    data. Usually, a dask worker will process a single partition. For multiple workers, this easily
    builds up to parallelized execution on multiple partitions at once.
 2. Build the DAG graph by calling dask-compatible libraries on your data.
 3. Call `dask.compute()` on your final object. This is the time where the execution happens.

Note: Spark is another library that does the same thing but is proprietary to Apache.

## Awkward


The [awkward](https://awkward-array.org/doc/main/) library is an extension of
numpy that works with irregularly shaped data commonly found in
HEP. It interfaces nicely with the `uproot` [docs](https://uproot.readthedocs.io/en/latest/uproot.html)
and can be used to manipulate root file data in python. It is not super fast but it can easily be
parallelized in our case since events are usually independent.

After some consideration, using Awkward as the data structure will at first be annoying because one
has to manipulate fields (and worst of all, nested fields). It will however pay off when using pytorch
`NestedTensor`s, since we can directly push root file (or parquet files) to pytorch Tensors without
padding. This is specially useful for graph or transformers that do not rely on padded or masking.

Since we mostly do not use awkward in our codebase, we will focus on dask_awkward instead.

## Dask + Awkward



The promises from [dask_awkward](https://dask-awkward.readthedocs.io/en/stable/) is that we can work
with almost all types of data in nested and irregularly shaped arrays. This makes it the most flexible
configuration for HEP data. The dask wrapper on top ensures that we can harness the benefits of
parallelized execution and also better resource management.

Some caveats:
 - Nested array (as awkward Records) are very nice to work with sine you can access sub-fields using
    the python attribute interface. Say you have an array like:

    ```python
    {"Lepton": {"pt": Array, "eta": Array, "phi": Array}}
    ```

    This means that you can access the field `"Lepton.pt"` as `array.Lepton.pt`

    The problem for us is that we do not know exactly the shape of the array nor do we want to hardcode
    the names of the fields into our code. Instead, we have two options:

    1. Access with tuple style: `array[("Lepton", "pt")]`. This works very well but can break some
        dask operations if one is not careful.
    2. Access with custom `NestedArrayIndexer.get_nested_array(array, "Lepton.pt")` method. This recursive
        calls `getattr` on the field until it finds the desired Array. This class can be found in
        `needle.etl.array`.
 - Do not mix dask with non-dask operations. Adding a scalar to a dask array might trigger unwanted
    DAG execution.
 - When using dask functions like `dask.map_partitions`, you are not guaranteed to get the information
    about how many events are in each partition. This can make it difficult to slice arrays in this way.
    See more in the section about KFold slicing.

Useful methods:
 - The events per partition are accessible from the array using `array.divisions`. This is a cumulative
    sum of the event index.
 - Sub-arrays per partitions are `array.partitions`
 - The total number of partitions is `array.npartitions`
 - Trigger computation using `array.compute`.

**Example**

```python
import dask_awkward as dak

array = dak.read_parquet("data.parquet")  # read as delayed with dask reader
mean = dak.mean(array)
std = dak.std(array)
normed_array = (array - mean) / std  # only use delayed objects
normed_array.compute()  # trigger computation
```

Note: Some functions from the awkward library are not implemented in dask_awkward. The solution is
simple: execute the DAG and apply awkward per field or for all fields but with taking care of the
computational overhead. For example by partition using `map_partitions` or `dask.delayed` for python
functions. Requires a little bit of getting used to it.

## Implementation of KFold Slicing

Slicing is tricky since we want it to be exact. Just picking 80% of the partitions will not do since
partitions are usually not all of the same size. Slicing like `array[0:8000]` for a 10k event array
would work but then at compute time, all the array must be read into memory and then sliced. This
is fine from a memory standpoint since dask takes care of not loading more partitions than fit into
memory. The issue is the IO, since you read a lot of unnecessary data that you might not use in the
end.

The padded IterableDatasets found in `ml/data` make use of partitions to efficiently distribute the
data between torch or dask workers. Slicing then occurs by handling the `array.divisions` metadata
object beforehand and determining which partitions have to be slicing and at which event.

Steps:

 1. Compute which events are part of the training fold (N-1) and which are part of the evaluation
   fold (1). These do not use percentages, but using 5 folds automatically implies a 80% training to
   evaluation ratio.
 2. Loop over valid partitions in the IterableDataset and potentially slice and array. This can be
   done with the :prop:`ml.data.KFold.partitions` dict. The slicing index is either
    - None: Do not slice this partition
    - int > 0: Slice until this event number
    - int < 0. Slice from this event on
    More info in the corresponding docstring.
 3. Finally, for simplicity, use the :func:`ml.data.io.load_partition()` function that does this
   automatically for you.
