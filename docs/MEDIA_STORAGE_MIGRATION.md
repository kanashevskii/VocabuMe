# Media storage migration plan

## Current boundary

`vocab.media_storage.MediaStorage` is the supported boundary for new media integrations. The initial `LocalMediaStorage` implementation deliberately preserves existing project-relative values such as `media/card_images/...`; it validates that reads stay within approved media roots.

Existing media is not moved or rewritten by this change.

## Future object-storage rollout

1. Add an object-storage implementation behind `MediaStorage`, with credentials supplied only through production secrets.
2. Introduce dual-read: object storage first, local filesystem fallback. Keep database path values unchanged.
3. Copy media in idempotent batches, recording checksums, byte counts, failures, and retry state. Do not delete local files.
4. Validate sampled and user-reported assets through the served HTTP endpoints, including non-empty bytes and correct media types.
5. Switch new writes after a documented rollback window. Retain local fallback until a restore drill proves recovery.
6. Obtain explicit approval before deleting any legacy objects or changing retention.
