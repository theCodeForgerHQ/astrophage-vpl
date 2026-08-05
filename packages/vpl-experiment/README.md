# vpl-experiment

The experiment manifest engine and the `vpl` command line — doc 08 §6, doc 11 §2 WBS 1.3.

One file, one experiment:

```bash
vpl run experiments/b02-reference-operating-point.yaml
vpl reproduce <run-id>          # re-executes from the archived manifest and verifies
vpl compare <run-id-a> <run-id-b>
```

Gate **G-1.3** (doc 11 §2) is "manifest reproduction bit-identical", and it is what
`vpl reproduce` exists to prove. See `docs/adr/ADR-008-manifest-substrate.md` for why the
configuration substrate is OmegaConf now and Hydra later, and
`vpl.experiment.digest` for what "bit-identical" is taken to mean when every artifact
also embeds the wall-clock time at which it was written.
