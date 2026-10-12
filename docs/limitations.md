# Limitations

- No verified real-data/paid-API reproduction result. Mock scores reflect synthetic
  candidate order and validate the plumbing only.
- Both upstream code licenses and original data redistribution rights are unknown.
  Do not copy upstream code or real raw data into this repository.
- Calendar timestamps are absent from the inspected raw schema. The converter
  explicitly uses sequence-only validation. Strict-time datasets require timezone
  timestamps and enforce both local-history and global collaborative-graph cutoffs.
- Warmup uses one audited round and sequential writes. It does not claim equivalence
  to the original trainer's parallel/asynchronous update schedule.
- Response replay is run-local. Resume rejects config/code/data changes. A response
  cached before a case commit is reusable; a process killed between dispatch and
  response persistence has an unknown billing outcome. Durable inflight markers
  block automatic redispatch. Resolve with provider-side records before clearing a
  marker; there is no automatic exactly-once guarantee from external APIs.
- A timeout retry can incur cost; every explicit retry counts toward max_calls, and
  unresolved/error reservations remain in conservative accounting. Billing is an
  estimate, not an invoice. Missing usage/prices are disclosed separately.
- Raw traces contain private user text locally. Outputs/cache/raw data are ignored
  by Git; only synthetic fixtures and sanitized summaries should be shared. POSIX
  permission 0600 is attempted; Windows uses the workspace's inherited ACL, so
  configure access restrictions separately for private datasets.
- Pricing has no guessed live defaults. Azure deployment and provider snapshot
  remain unknown in example configs; replace them before controlled real runs.
- Running the CLI from a source checkout is the supported distribution. Upstream
  checkouts/configs are external resources; a standalone wheel is not self-contained.
