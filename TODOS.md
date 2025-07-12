 - [ ] Fix setup.sh script returning Success even if law failed
 - [ ] Chunked dataset is shuffled at instantiation but this prevents the validation
    step from running on a non-shuffled version. This is not such a big issue
    as we should split the datasets anyways.
 - [ ] How to split the four types of datasets (train, validate, test and predict)
