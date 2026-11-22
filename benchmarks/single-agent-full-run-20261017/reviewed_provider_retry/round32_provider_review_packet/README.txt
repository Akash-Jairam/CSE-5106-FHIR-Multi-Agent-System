Round32 provider-review metadata for the two reviewed baseline tasks.
Both tasks were rejected again by Azure HTTP400 content_filter /
ResponsibleAIPolicyViolation with jailbreak detected=true and filtered=true.
Both now have three recorded rejections and remain pending. No fourth request
is authorized by this packet. No provider messages, prompts, clinical content,
answers, reference values or credentials are included.

Each task made one successful call followed by a rejected call. Request-ID
associations are inferred only: each owned worker has a unique successful
prompt/completion numeric prefix followed by counter2 HTTP400. There is no
direct logged question-ID/request-ID join; best-effort logging limits remain.
Cached counts are excluded from match keys. Journal hashes were stable after
the batch seal but are not covered by that seal. Original prior review IDs
and their inference limits remain in the earlier packet; they are not replaced.

All failed-task time/tokens/known successful-call cost stay outside primary
accepted results. Rejected-call usage and actual provider billing are unknown.
No calls, native checks, scoring, inference changes or safeguard changes were
performed by this metadata review. See incident.json for safe IDs and UTCs.
