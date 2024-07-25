# Algo Analysis project

# To run tests

```
cd internal/openfold/dev_team/algo_analysis
make test
```

## Key Points

Wherever possible, embedding vectors from different attention heads are concatendated to extend the embedding axis, rather than creating a new axis
that indexes the heads.
