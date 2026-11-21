# Security

## Supported versions

Only the latest commit on `main` is supported. There are no releases and no backports.

## What to expect from the server

`engine serve` has no authentication, no TLS and no rate limit, and it binds to `127.0.0.1` by default for that reason. Anyone who can reach its port can use the GPU and read every request's events on `/v1/steps`. Do not expose it publicly, forward its port or tunnel it. That is by design and is not a vulnerability; see [reproduce.md](reproduce.md#running).

## Reporting a vulnerability

Please do not open a public issue for a security problem. Report it privately through GitHub: on the repository's **Security** tab choose **Report a vulnerability**. Say what is affected (the engine, the HTTP server, the scripts or the site in `web/`), the commit, the steps to reproduce it and what an attacker gains.

This is a one-person project. Expect a first answer within a week, and a fix or a decision not to fix after that, with credit in the changelog if you want it.

Things worth reporting: memory safety bugs reachable from a request or from a model file (the safetensors header parser, the tokenizer, the JSON bodies of the HTTP API), a way for one request to read another request's tokens, and vulnerable dependencies of the site.
