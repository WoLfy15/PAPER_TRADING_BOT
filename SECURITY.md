# Security notes (v3)

## What was found in the uploaded archive, and what was done about it

The original `alpaca_vol_agent_v2.zip` contained a real `.env` file with
populated `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` values.

**What this means, plainly:** those two values are real Alpaca API
credentials, and this project's own code (`.env`'s companion
`I_UNDERSTAND_THIS_IS_LIVE`/`DRY_RUN` flags) documents that a "fresh
paper account created for this event" was the intent — and the key ID's
`PK` prefix is Alpaca's own convention for **paper** trading keys, not
live ones. So this is very likely a paper-trading key, not one that can
move real money. That said, treat it as compromised regardless:

- **It left your machine.** The moment a credential is included in an
  upload to any external system — this one included — the only safe
  assumption is that it is no longer private to you alone.
- **Paper keys still grant real API access** to your Alpaca paper
  account (placing orders, reading account data). That's a smaller
  blast radius than a live key, but it's not nothing.

### What was checked

- `.env` was confirmed **never committed to this project's git
  history** (checked directly: `git log --all --diff-filter=A
  --name-only -- '*.env'` and a full-history file-list grep both came
  back empty). There is no git-history purge needed.
- The pre-existing `.gitignore` already correctly excludes `.env` and
  `agent_state.json` from version control — that part was already right.

### What this delivery does

- **`.env` is not included in this delivered archive at all** — only
  `.env.example` (placeholders, no real values) ships.
- The real key values were never echoed back in full anywhere in this
  conversation or in any file produced here — only variable *names*
  and, for the safety check itself, non-secret metadata (field
  lengths, the `PK` prefix) were inspected.

### What you still need to do

1. **Rotate the key anyway.** Log into
   [app.alpaca.markets](https://app.alpaca.markets), go to your paper
   account's API keys section, revoke the existing key pair, and
   generate a new one. This is a two-minute task and costs nothing —
   there's no reason not to, given the key already left your machine
   once.
2. Put the **new** key into your own local `.env` (copy
   `.env.example` to `.env` and fill it in) — never into a file you
   intend to upload, paste, or commit anywhere.
3. If you ever *do* need to share this project (a bug report, a
   teammate, another AI tool), either delete `.env` first or use
   something like `git archive` / a fresh `git clone` into a temp
   directory, which naturally excludes anything `.gitignore`d.

## Ongoing hygiene this project follows

- `.gitignore` excludes `.env`, `agent_state.json`, and (already,
  before v3) a stray nested-clone directory.
- `config.py` only ever reads credentials from environment variables —
  nothing hardcodes a fallback secret anywhere.
- `alpaca_client.py` never logs the credential headers it sends
  (`log.debug` calls in that module log the method/URL/status, never
  the request headers).
- New in v3: `execution/reconciliation.py` and the dashboard's `/orders`
  endpoint only ever report on orders this project's own
  `client_order_id` prefix (`volagent-...`) identifies as its own —
  they never touch or expose any other activity on the same account.

## A general note on this class of finding

Finding real credentials sitting in a project upload is common — it
usually means a `.env` file (correctly `.gitignore`d for version
control) got swept up anyway when zipping a whole project directory
for some other purpose. The fix isn't just "remove it from this one
delivery" — it's rotating the credential, because zipping-and-sharing
is exactly the kind of one-off action that's easy to repeat by
habit before remembering to exclude it again.
