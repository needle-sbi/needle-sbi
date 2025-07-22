 - [ ] Fix setup.sh script returning Success even if law failed
 - [ ] Chunked dataset is shuffled at instantiation but this prevents the validation
    step from running on a non-shuffled version. This is not such a big issue
    as we should split the datasets anyways.
 - [ ] How to split the four types of datasets (train, validate, test and predict)

 - Configurations that work:
   1. ParticleChunked (pytorch multiprocessing)
      1.1 split_row_groups=True on
         1.1.1 With 4 CF nested Records parquet files
            [x] With 10 workers
            [x] With 1 worker
         1.1.2 With 1 Root->Parquet converted DELPHES sample
            [ ] With 1 worker. 
